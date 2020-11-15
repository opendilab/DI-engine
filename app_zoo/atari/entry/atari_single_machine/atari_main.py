import argparse
import copy
from collections import OrderedDict

from nervex.entry.base_single_machine import SingleMachineRunner
from nervex.worker.agent import DqnActorAgent, ACActorAgent, DiscreteEvaluatorAgent, ACDiscreteEvaluatorAgent
from nervex.utils import read_config
from nervex.worker import SubprocessEnvManager
from app_zoo.atari.envs import AtariEnv
from app_zoo.atari.worker import AtariDqnLearner, AtariPpoLearner


class AtariRunner(SingleMachineRunner):

    def _setup_env(self):
        actor_env_num = self.cfg.actor.env_num
        actor_env_cfg = copy.deepcopy(self.cfg.env)
        actor_env_cfg.is_train = True
        self.actor_env = SubprocessEnvManager(
            AtariEnv,
            env_cfg=[actor_env_cfg for _ in range(actor_env_num)],
            env_num=actor_env_num,
            episode_num=self.cfg.actor.episode_num
        )

        eval_env_num = self.cfg.evaluator.env_num
        evaluate_env_cfg = copy.deepcopy(self.cfg.env)
        evaluate_env_cfg.is_train = False
        self.evaluate_env = SubprocessEnvManager(
            AtariEnv,
            env_cfg=[evaluate_env_cfg for _ in range(eval_env_num)],
            env_num=eval_env_num,
            episode_num=self.cfg.evaluator.episode_num
        )

    def _setup_learner(self):
        if self.algo_type == 'dqn':
            self.learner = AtariDqnLearner(self.cfg)
        elif self.algo_type == 'ppo':
            self.learner = AtariPpoLearner(self.cfg)

    def _setup_agent(self):
        if self.algo_type == 'dqn':
            self.actor_agent = DqnActorAgent(copy.deepcopy(self.learner.agent.model))
            self.evaluator_agent = DiscreteEvaluatorAgent(copy.deepcopy(self.learner.agent.model))
        elif self.algo_type == 'ppo':
            self.actor_agent = ACActorAgent(copy.deepcopy(self.learner.agent.model))
            self.evaluator_agent = ACDiscreteEvaluatorAgent(copy.deepcopy(self.learner.agent.model))
            print(self.evaluator_agent, self.actor_agent)
        self.actor_agent.mode(train=False)
        self.evaluator_agent.mode(train=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default="./atari_dqn_default_config.yaml")
    args = parser.parse_args()
    runner = AtariRunner(read_config(args.config_path))
    runner.run()
