import os
import gym
import numpy as np
import copy
import torch
from tensorboardX import SummaryWriter
from functools import partial

from ding.config import compile_config
from ding.worker import BaseLearner, Episode1v1Collector, NaiveReplayBuffer, BaseSerialEvaluator
from ding.envs import BaseEnvManager
from ding.policy import PPOPolicy
from ding.model import VAC
from ding.utils import set_pkg_seed
from dizoo.slime_volley.envs import SlimeVolleyEnv
from dizoo.slime_volley.config import slime_volley_league_ppo_config



def main(cfg, seed=0, max_iterations=int(1e10)):
    cfg.exp_name = 'slime_volley_selfplay_ppo'
    cfg = compile_config(
        cfg,
        BaseEnvManager,
        PPOPolicy,
        BaseLearner,
        Episode1v1Collector,
        BaseSerialEvaluator,
        NaiveReplayBuffer,
        save_cfg=True
    )
    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    collector_env_cfg = copy.deepcopy(cfg.env)
    collector_env_cfg.is_evaluator = False
    evaluator_env_cfg = copy.deepcopy(cfg.env)
    evaluator_env_cfg.is_evaluator = True
    collector_env = BaseEnvManager(env_fn=[partial(SlimeVolleyEnv, collector_env_cfg) for _ in range(collector_env_num)], cfg=cfg.env.manager)
    evaluator_env = BaseEnvManager(env_fn=[partial(SlimeVolleyEnv, evaluator_env_cfg) for _ in range(evaluator_env_num)], cfg=cfg.env.manager)

    collector_env.seed(seed)
    evaluator_env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

    model1 = VAC(**cfg.policy.model)
    policy1 = PPOPolicy(cfg.policy, model=model1)
    model2 = VAC(**cfg.policy.model)
    policy2 = PPOPolicy(cfg.policy, model=model2)

    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    learner1 = BaseLearner(
        cfg.policy.learn.learner, policy1.learn_mode, tb_logger, exp_name=cfg.exp_name, instance_name='learner1'
    )
    learner2 = BaseLearner(
        cfg.policy.learn.learner, policy2.learn_mode, tb_logger, exp_name=cfg.exp_name, instance_name='learner2'
    )
    collector = Episode1v1Collector(
        cfg.policy.collect.collector,
        collector_env, [policy1.collect_mode, policy2.collect_mode],
        tb_logger,
        exp_name=cfg.exp_name
    )
    evaluator_cfg = copy.deepcopy(cfg.policy.eval.evaluator)
    evaluator_cfg.stop_value = cfg.env.stop_value
    evaluator = BaseSerialEvaluator(
        evaluator_cfg,
        evaluator_env,
        policy1.eval_mode,
        tb_logger,
        exp_name=cfg.exp_name,
        instance_name='builtin_ai_evaluator'
    )

    stop_flag = False
    for _ in range(max_iterations):
        if evaluator.should_eval(learner1.train_iter):
            stop_flag, reward = evaluator.eval(learner1.save_checkpoint, learner1.train_iter, collector.envstep)
            tb_logger.add_scalar('fixed_evaluator_step/reward_mean', reward, collector.envstep)
        if stop_flag:
            break
        train_data, _ = collector.collect(train_iter=learner1.train_iter)
        for data in train_data:
            for d in data:
                d['adv'] = d['reward']
        for i in range(cfg.policy.learn.update_per_collect):
            learner1.train(train_data[0], collector.envstep)
            learner2.train(train_data[1], collector.envstep)


if __name__ == "__main__":
    main(slime_volley_league_ppo_config)
