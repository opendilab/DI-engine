import time
import argparse
import torch

from nervex.worker import SubprocessEnvManager
from nervex.worker.agent.sumo_dqn_agent import SumoDqnActorAgent
from nervex.entry.base import SingleMachineRunner
from nervex.envs.sumo.sumo_env import SumoWJ3Env
from nervex.worker.learner.sumo_dqn_learner import SumoDqnLearner
from nervex.utils import read_config


class SumoRunner(SingleMachineRunner):
    def _setup_env(self):
        env_num = self.cfg.env.env_num
        self.env = SubprocessEnvManager(SumoWJ3Env, env_cfg=[self.cfg.env for _ in range(env_num)], env_num=env_num)

    def _setup_learner(self):
        self.learner = SumoDqnLearner(self.cfg)

    def _setup_actor_agent(self):
        self.actor_agent = SumoDqnActorAgent(self.learner.agent.model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default="./sumo_dqn_default_config.yaml")
    args = parser.parse_known_args()[0]
    runner = SumoRunner(read_config(args.config_path))
    runner.run()
