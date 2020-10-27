import argparse
import copy

from nervex.entry.base_single_machine import SingleMachineRunner
from nervex.utils import read_config
from nervex.worker import SubprocessEnvManager
from app_zoo.sumo.envs import SumoWJ3Env, FakeSumoWJ3Env
from app_zoo.sumo.worker.agent.sumo_dqn_agent import SumoDqnActorAgent, SumoDqnEvaluateAgent
from app_zoo.sumo.worker.learner.sumo_dqn_learner import SumoDqnLearner


class SumoRunner(SingleMachineRunner):

    def _setup_env(self):
        env_num = self.cfg.env.env_num
        self.env = SubprocessEnvManager(SumoWJ3Env, env_cfg=[self.cfg.env for _ in range(env_num)], env_num=env_num)

    def _setup_learner(self):
        self.learner = SumoDqnLearner(self.cfg)

    def _setup_actor_agent(self):
        self.actor_agent = SumoDqnActorAgent(copy.deepcopy(self.learner.agent.model))

    def _setup_evaluate_agent(self):
        self.evaluate_agent = SumoDqnEvaluateAgent(copy.deepcopy(self.learner.agent.model))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default="./sumo_dqn_default_config.yaml")
    args = parser.parse_known_args()[0]
    runner = SumoRunner(read_config(args.config_path))
    runner.run()
