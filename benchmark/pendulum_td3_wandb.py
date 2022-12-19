import gym
from ditk import logging
from ding.model.template.qac import QAC
from ding.policy import TD3Policy
from ding.envs import DingEnvWrapper, BaseEnvManagerV2, SubprocessEnvManagerV2
from ding.data import DequeBuffer
from ding.config import compile_config
from ding.framework import task
from ding.framework.context import OnlineRLContext
from ding.framework.middleware import OffPolicyLearner, StepCollector, interaction_evaluator, data_pusher, CkptSaver, wandb_online_logger
from ding.utils import set_pkg_seed
from ding.utils.log_helper import build_logger
from dizoo.classic_control.pendulum.envs.pendulum_env import PendulumEnv

from pendulum_td3_config import main_config, create_config
import wandb


def main(seed):
    logging.getLogger().setLevel(logging.INFO)
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)
    cfg.env.seed = seed

    wandb.init(
        # Set the project where this run will be logged
        project='pendulum-td3-0',
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=f"td3",
        # Track hyperparameters and run metadata
        config=cfg
    )

    logger_, tb_logger = build_logger(path='./log/pendulum_td3/seed' + str(seed), need_tb=True)
    with task.start(async_mode=False, ctx=OnlineRLContext()):
        collector_env = SubprocessEnvManagerV2(
            env_fn=[lambda: PendulumEnv(cfg.env) for _ in range(cfg.env.collector_env_num)], cfg=cfg.env.manager
        )
        evaluator_env = SubprocessEnvManagerV2(
            env_fn=[lambda: PendulumEnv(cfg.env) for _ in range(cfg.env.evaluator_env_num)], cfg=cfg.env.manager
        )
        evaluator_env.enable_save_replay(replay_path=cfg.policy.logger.record_path)

        set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

        model = QAC(**cfg.policy.model)
        buffer_ = DequeBuffer(size=cfg.policy.other.replay_buffer.replay_buffer_size)
        policy = TD3Policy(cfg.policy, model)

        task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
        task.use(
            StepCollector(cfg, policy.collect_mode, collector_env, random_collect_size=cfg.policy.random_collect_size)
        )
        task.use(data_pusher(cfg, buffer_))
        task.use(OffPolicyLearner(cfg, policy.learn_mode, buffer_))
        task.use(wandb_online_logger(cfg.policy.logger, evaluator_env, model))
        task.use(CkptSaver(cfg, policy, train_freq=100))
        task.run()


if __name__ == "__main__":
    main(0)
