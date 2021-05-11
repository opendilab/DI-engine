import os
import gym
from tensorboardX import SummaryWriter

from nervex.config import compile_config
from nervex.worker import BaseLearner, BaseSerialCollector, BaseSerialEvaluator
from nervex.data import BufferManager
from nervex.envs import BaseEnvManager, NervexEnvWrapper
from nervex.policy import DDPGPolicy
from nervex.model import QAC
from nervex.utils import set_pkg_seed
from app_zoo.classic_control.pendulum.envs import PendulumEnv
from app_zoo.classic_control.pendulum.config.pendulum_td3_config import pendulum_td3_config


def main(cfg, seed=0):

    def wrapped_pendulum_env():
        return NervexEnvWrapper(gym.make('Pendulum-v0'), cfg=cfg.env.wrapper)

    cfg = compile_config(
        cfg,
        PendulumEnv,
        BaseEnvManager,
        DDPGPolicy,
        BaseLearner,
        BaseSerialCollector,
        BaseSerialEvaluator,
        BufferManager,
        save_cfg=True
    )

    # Set up envs for collection and evaluation
    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    collector_env = BaseEnvManager(
        env_fn=[lambda: PendulumEnv(cfg.env) for _ in range(collector_env_num)], cfg=cfg.env.manager
    )
    evaluator_env = BaseEnvManager(
        env_fn=[lambda: PendulumEnv(cfg.env) for _ in range(evaluator_env_num)], cfg=cfg.env.manager
    )

    # Set random seed for all package and instance
    collector_env.seed(seed)
    evaluator_env.seed(seed)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

    # Set up RL Policy
    model = QAC(**cfg.policy.model)
    policy = DDPGPolicy(cfg.policy, model=model)

    # Set up collection, training and evaluation utilities
    tb_logger = SummaryWriter(os.path.join('./log/', 'serial'))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger)
    collector = BaseSerialCollector(cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger)
    evaluator = BaseSerialEvaluator(cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger)
    replay_buffer = BufferManager(cfg.policy.other.replay_buffer, tb_logger)

    # Training & Evaluation loop
    while True:
        # Evaluate at the beginning and with specific frequency
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break
        # Collect data from environments
        new_data = collector.collect_data(learner.train_iter)
        replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        # Trian
        for i in range(cfg.policy.learn.update_per_collect):
            train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
            if train_data is not None:
                learner.train(train_data, collector.envstep)


if __name__ == "__main__":
    main(pendulum_td3_config, seed=0)
