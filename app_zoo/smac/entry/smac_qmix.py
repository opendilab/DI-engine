import argparse
import torch
import copy
from collections import OrderedDict

from nervex.entry.base_single_machine import SingleMachineRunner
from nervex.model import QMix
from nervex.data import timestep_collate, AsyncDataLoader
from nervex.utils import read_config
from nervex.worker import BaseLearner, SubprocessEnvManager
from nervex.worker.agent import create_qmix_learner_agent, create_qmix_actor_agent, create_qmix_evaluator_agent
from app_zoo.smac.envs import FakeSMACEnv, SMACEnv
from app_zoo.smac.computation_graph import SMACQMixGraph


class SMACQMixLearner(BaseLearner):
    _name = "SMACQMixLearner"

    def _setup_agent(self):
        env_info = SMACEnv().info()
        # dim -1 means ignore agent num
        model = QMix(
            env_info.agent_num, env_info.obs_space.shape['agent_state'][-1], env_info.obs_space.shape['global_state'],
            env_info.act_space.shape[-1], self._cfg.model.embedding_dim
        )
        if self._cfg.learner.use_cuda:
            model.cuda()
        self._agent = create_qmix_learner_agent(
            model, state_num=self._cfg.learner.data.batch_size, agent_num=self._cfg.env.agent_num
        )
        self._agent.mode(train=True)
        self._agent.target_mode(train=True)

    def _setup_computation_graph(self):
        self._computation_graph = SMACQMixGraph(self._cfg.learner)

    def _setup_dataloader(self):
        cfg = self._cfg.learner.data
        self._dataloader = AsyncDataLoader(
            self.get_data, cfg.batch_size, self._device, cfg.chunk_size, timestep_collate, cfg.num_workers
        )


class SMACQMixRunner(SingleMachineRunner):

    def _setup_env(self):
        actor_env_num = self.cfg.actor.env_num
        actor_env_cfg = copy.deepcopy(self.cfg.env)
        self.actor_env = SubprocessEnvManager(
            SMACEnv,
            env_cfg=[actor_env_cfg for _ in range(actor_env_num)],
            env_num=actor_env_num,
            episode_num=self.cfg.actor.episode_num
        )
        self.actor_env.launch()

        eval_env_num = self.cfg.evaluator.env_num
        evaluate_env_cfg = copy.deepcopy(self.cfg.env)
        self.evaluate_env = SubprocessEnvManager(
            SMACEnv,
            env_cfg=[evaluate_env_cfg for _ in range(eval_env_num)],
            env_num=eval_env_num,
            episode_num=self.cfg.evaluator.episode_num
        )
        self.evaluate_env.launch()

    def _setup_learner(self):
        self.learner = SMACQMixLearner(self.cfg)

    def _setup_agent(self):
        self.actor_agent = create_qmix_actor_agent(
            copy.deepcopy(self.learner.agent.model), state_num=self.cfg.actor.env_num, agent_num=self.cfg.env.agent_num
        )
        self.actor_agent.mode(train=False)
        self.evaluator_agent = create_qmix_evaluator_agent(
            copy.deepcopy(self.learner.agent.model),
            state_num=self.cfg.evaluator.env_num,
            agent_num=self.cfg.env.agent_num
        )
        self.evaluator_agent.mode(train=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default="./smac_qmix_default_config.yaml")
    args = parser.parse_known_args()[0]
    runner = SMACQMixRunner(read_config(args.config_path))
    runner.run()
