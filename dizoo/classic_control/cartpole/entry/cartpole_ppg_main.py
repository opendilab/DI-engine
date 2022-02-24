import os
import gym
from tensorboardX import SummaryWriter
from easydict import EasyDict
from copy import deepcopy

from ding.config import compile_config
from ding.worker import BaseLearner, SampleSerialCollector, InteractionSerialEvaluator, AdvancedReplayBuffer
from ding.envs import BaseEnvManager, DingEnvWrapper
from ding.policy import PPGPolicy
from ding.model import PPG
from ding.utils import set_pkg_seed, deep_merge_dicts
from dizoo.classic_control.cartpole.config.cartpole_ppg_config import cartpole_ppg_config


def wrapped_cartpole_env():
    return DingEnvWrapper(
        gym.make('CartPole-v0'),
        EasyDict(env_wrapper='default'),
    )


def main(cfg, seed=0, max_train_iter=int(1e8), max_env_step=int(1e8)):
    cfg = compile_config(
        cfg,
        BaseEnvManager,
        PPGPolicy,
        BaseLearner,
        SampleSerialCollector,
        InteractionSerialEvaluator, {
            'policy': AdvancedReplayBuffer,
            'value': AdvancedReplayBuffer
        },
        save_cfg=True
    )
    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    collector_env = BaseEnvManager(env_fn=[wrapped_cartpole_env for _ in range(collector_env_num)], cfg=cfg.env.manager)
    evaluator_env = BaseEnvManager(env_fn=[wrapped_cartpole_env for _ in range(evaluator_env_num)], cfg=cfg.env.manager)

    collector_env.seed(seed)
    evaluator_env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

    model = PPG(**cfg.policy.model)
    policy = PPGPolicy(cfg.policy, model=model)
    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    collector = SampleSerialCollector(
        cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger, exp_name=cfg.exp_name
    )
    evaluator = InteractionSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )
    policy_buffer = AdvancedReplayBuffer(
        cfg.policy.other.replay_buffer.policy, tb_logger, exp_name=cfg.exp_name, instance_name='policy_buffer'
    )
    value_buffer = AdvancedReplayBuffer(
        cfg.policy.other.replay_buffer.value, tb_logger, exp_name=cfg.exp_name, instance_name='value_buffer'
    )

    while True:
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break
        new_data = collector.collect(train_iter=learner.train_iter)
        policy_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        value_buffer.push(deepcopy(new_data), cur_collector_envstep=collector.envstep)
        for i in range(cfg.policy.learn.update_per_collect):
            batch_size = learner.policy.get_attribute('batch_size')
            policy_data = policy_buffer.sample(batch_size['policy'], learner.train_iter)
            value_data = value_buffer.sample(batch_size['value'], learner.train_iter)
            if policy_data is not None and value_data is not None:
                train_data = {'policy': policy_data, 'value': value_data}
                learner.train(train_data, collector.envstep)
        policy_buffer.clear()
        value_buffer.clear()
        if learner.train_iter >= max_train_iter or collector.envstep >= max_env_step:
            break


if __name__ == "__main__":
    main(cartpole_ppg_config)
