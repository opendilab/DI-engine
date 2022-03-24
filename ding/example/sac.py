import gym
import logging
from ding.model import QAC
from ding.policy import SACPolicy
from ding.envs import DingEnvWrapper, BaseEnvManager
from ding.data import DequeBuffer
from ding.config import compile_config
from ding.framework import task
from ding.framework.context import OnlineRLContext
from ding.framework.middleware import offpolicy_data_fetcher, trainer, StepCollector, interaction_evaluator, \
    CkptSaver
from ding.utils import set_pkg_seed
from dizoo.classic_control.pendulum.config.pendulum_sac_config import main_config, create_config


def main():
    logging.getLogger().setLevel(logging.INFO)
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)
    with task.start(async_mode=False, ctx=OnlineRLContext()):
        collector_env = BaseEnvManager(
            env_fn=[lambda: DingEnvWrapper(gym.make("Pendulum-v0")) for _ in range(8)], cfg=cfg.env.manager
        )
        evaluator_env = BaseEnvManager(
            env_fn=[lambda: DingEnvWrapper(gym.make("Pendulum-v0")) for _ in range(5)], cfg=cfg.env.manager
        )

        set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

        model = QAC(**cfg.policy.model)
        buffer_ = DequeBuffer(size=cfg.policy.other.replay_buffer.replay_buffer_size)
        policy = SACPolicy(cfg.policy, model=model)

        task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
        task.use(StepCollector(cfg, policy.collect_mode, collector_env))
        task.use(offpolicy_data_fetcher(cfg, buffer_, random_collect_size=1000))
        task.use(trainer(cfg, policy.learn_mode))
        task.use(CkptSaver(cfg, policy, train_freq=100))
        task.run(max_step=100000)


if __name__ == "__main__":
    main()
