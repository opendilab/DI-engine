import argparse
import copy

from nervex.entry.base_single_machine import SingleMachineRunner
from nervex.utils import read_config
from nervex.worker import SubprocessEnvManager
from nervex.worker.agent import DqnActorAgent, DiscreteEvaluatorAgent
from app_zoo.sumo.envs import SumoWJ3Env, FakeSumoWJ3Env
from app_zoo.sumo.worker.learner.sumo_dqn_learner import SumoDqnLearner


class SumoRunner(SingleMachineRunner):

    def _setup_env(self):
        actor_env_num = self.cfg.actor.env_num
        eval_env_num = self.cfg.evaluator.env_num
        self.actor_env = SubprocessEnvManager(
            SumoWJ3Env,
            env_cfg=[self.cfg.env for _ in range(actor_env_num)],
            env_num=actor_env_num,
            episode_num=self.cfg.actor.episode_num
        )
        self.evaluate_env = SubprocessEnvManager(
            SumoWJ3Env,
            env_cfg=[self.cfg.env for _ in range(eval_env_num)],
            env_num=eval_env_num,
            episode_num=self.cfg.evaluator.episode_num
        )

    def _setup_learner(self):
        self.learner = SumoDqnLearner(self.cfg)

    def _setup_agent(self):
        self.actor_agent = DqnActorAgent(copy.deepcopy(self.learner.agent.model))
        self.actor_agent.mode(train=False)
        self.evaluator_agent = DiscreteEvaluatorAgent(copy.deepcopy(self.learner.agent.model))
        self.evaluator_agent.mode(train=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default="./sumo_dqn_default_config.yaml")
    args = parser.parse_known_args()[0]
    runner = SumoRunner(read_config(args.config_path))
    runner.run()
