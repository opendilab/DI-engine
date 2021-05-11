import os
import gym
from tensorboardX import SummaryWriter
from easydict import EasyDict

from nervex.config import compile_config
from nervex.worker import BaseLearner, BaseSerialCollector, BaseSerialEvaluator
from nervex.data import BufferManager
from nervex.envs import BaseEnvManager, NervexEnvWrapper
from nervex.policy import PPOPolicy
from nervex.model import FCValueAC
from nervex.utils import set_pkg_seed, deep_merge_dicts
from app_zoo.classic_control.cartpole.envs import CartPoleEnv
from app_zoo.classic_control.cartpole.config.cartpole_ppo_config import cartpole_ppo_config


def wrapped_cartpole_env():
    return NervexEnvWrapper(gym.make('CartPole-v0'))


def main(cfg, seed=0):
    cfg = compile_config(cfg, CartPoleEnv, BaseEnvManager, PPOPolicy, BaseLearner, BaseSerialCollector, BaseSerialEvaluator, BufferManager, save_cfg=True)
    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    # collector_env = BaseEnvManager(env_fn=[wrapped_cartpole_env for _ in range(collector_env_num)], cfg=cfg.env.manager)
    # evaluator_env = BaseEnvManager(env_fn=[wrapped_cartpole_env for _ in range(evaluator_env_num)], cfg=cfg.env.manager)
    collector_env = BaseEnvManager(env_fn=[lambda: CartPoleEnv({}) for _ in range(collector_env_num)], cfg=cfg.env.manager)
    evaluator_env = BaseEnvManager(env_fn=[lambda: CartPoleEnv({}) for _ in range(evaluator_env_num)], cfg=cfg.env.manager)

    collector_env.seed(seed)
    evaluator_env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

    model = FCValueAC(**cfg.policy.model)
    policy = PPOPolicy(cfg.policy, model=model)
    tb_logger = SummaryWriter(os.path.join('./log/', 'serial'))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger)
    collector = BaseSerialCollector(cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger)
    evaluator = BaseSerialEvaluator(cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger)
    replay_buffer = BufferManager(cfg.policy.other.replay_buffer, tb_logger)

    while True:
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break
        new_data = collector.collect_data(learner.train_iter)
        replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        for i in range(cfg.policy.learn.update_per_collect):
            train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
            if train_data is not None:
                learner.train(train_data, collector.envstep)
        replay_buffer.clear()


if __name__ == "__main__":
    main(cartpole_ppo_config)
