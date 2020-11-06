import argparse
import copy
from collections import OrderedDict

from nervex.entry.base_single_machine import SingleMachineRunner
from nervex.utils import read_config
from nervex.worker import SubprocessEnvManager
from app_zoo.atari.envs import AtariEnv
from app_zoo.atari.worker.atari_agent import AtariActorAgent, AtariEvaluateAgent
from app_zoo.atari.worker.atari_dqn_learner import AtariDqnLearner


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
        self.learner = AtariDqnLearner(self.cfg)

    def _setup_agent(self):
        self.actor_agent = AtariActorAgent(copy.deepcopy(self.learner.agent.model))
        self.actor_agent.mode(train=False)
        self.evaluate_agent = AtariEvaluateAgent(copy.deepcopy(self.learner.agent.model))
        self.evaluate_agent.mode(train=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default="./atari_default_config.yaml")
    args = parser.parse_known_args()[0]
    runner = AtariRunner(read_config(args.config_path))
    runner.run()
