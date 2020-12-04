import argparse
import copy

from nervex.entry.base_single_machine import SingleMachineRunner
from nervex.utils import read_config
from nervex.worker import SubprocessEnvManager
from nervex.worker.agent import create_qac_evaluator_agent, create_qac_actor_agent
from app_zoo.classic_control.pendulum.envs import PendulumEnv
from app_zoo.classic_control.pendulum.worker import PendulumDdpgLearner


class SumoRunner(SingleMachineRunner):

    def _setup_env(self):
        actor_env_num = self.cfg.actor.env_num
        eval_env_num = self.cfg.evaluator.env_num
        self.actor_env = SubprocessEnvManager(
            PendulumEnv,
            env_cfg=[self.cfg.env for _ in range(actor_env_num)],
            env_num=actor_env_num,
            episode_num=self.cfg.actor.episode_num
        )
        self.actor_env.launch()
        self.evaluate_env = SubprocessEnvManager(
            PendulumEnv,
            env_cfg=[self.cfg.env for _ in range(eval_env_num)],
            env_num=eval_env_num,
            episode_num=self.cfg.evaluator.episode_num
        )
        self.evaluate_env.launch()

    def _setup_learner(self):
        self.learner = PendulumDdpgLearner(self.cfg)

    def _setup_agent(self):
        self.actor_agent = create_qac_actor_agent(copy.deepcopy(self.learner.agent.model))
        self.actor_agent.mode(train=False)
        self.evaluator_agent = create_qac_evaluator_agent(copy.deepcopy(self.learner.agent.model))
        self.evaluator_agent.mode(train=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default="./pendulum_ddpg_default_config.yaml")
    args = parser.parse_known_args()[0]
    runner = SumoRunner(read_config(args.config_path))
    runner.run()
