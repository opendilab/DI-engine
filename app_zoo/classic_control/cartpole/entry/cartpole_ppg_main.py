import os
import gym
from tensorboardX import SummaryWriter
from easydict import EasyDict
from copy import deepcopy

from nervex.config import compile_config
from nervex.worker import BaseLearner, SampleCollector, BaseSerialEvaluator, PrioritizedReplayBuffer
from nervex.envs import BaseEnvManager, NervexEnvWrapper
from nervex.policy import PPGPolicy
from nervex.model import FCPPG
from nervex.utils import set_pkg_seed, deep_merge_dicts
from app_zoo.classic_control.cartpole.config.cartpole_ppg_config import cartpole_ppg_config


def wrapped_cartpole_env():
    return NervexEnvWrapper(gym.make('CartPole-v0'))


def main(cfg, seed=0, max_iterations=int(1e10)):
    cfg = compile_config(
        cfg,
        BaseEnvManager,
        PPGPolicy,
        BaseLearner,
        SampleCollector,
        BaseSerialEvaluator,
        PrioritizedReplayBuffer,
        save_cfg=True
    )
    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    collector_env = BaseEnvManager(env_fn=[wrapped_cartpole_env for _ in range(collector_env_num)], cfg=cfg.env.manager)
    evaluator_env = BaseEnvManager(env_fn=[wrapped_cartpole_env for _ in range(evaluator_env_num)], cfg=cfg.env.manager)

    collector_env.seed(seed)
    evaluator_env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

    model = FCPPG(**cfg.policy.model)
    policy = PPGPolicy(cfg.policy, model=model)
    tb_logger = SummaryWriter(os.path.join('./log/', 'serial'))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger)
    collector = SampleCollector(cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger)
    evaluator = BaseSerialEvaluator(cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger)
    policy_buffer = PrioritizedReplayBuffer('policy_buffer', cfg.policy.other.replay_buffer.policy_buffer, tb_logger)
    value_buffer = PrioritizedReplayBuffer('value_buffer', cfg.policy.other.replay_buffer.value_buffer, tb_logger)

    while True:
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break
        new_data = collector.collect_data(learner.train_iter)
        policy_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        value_buffer.push(deepcopy(new_data), cur_collector_envstep=collector.envstep)
        for i in range(cfg.policy.learn.update_per_collect):
            policy_data = policy_buffer.sample(
                learner.policy.get_attribute('batch_size')['policy'], learner.train_iter)
            value_data = policy_buffer.sample(
                learner.policy.get_attribute('batch_size')['value'], learner.train_iter)
            if policy_data is not None and value_data is not None:
                train_data = {'policy': policy_data, 'value': value_data}
                learner.train(train_data, collector.envstep)
        policy_buffer.clear()
        value_buffer.clear()


if __name__ == "__main__":
    main(cartpole_ppg_config)
