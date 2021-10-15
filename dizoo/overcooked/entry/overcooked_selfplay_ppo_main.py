import os
import gym
import numpy as np
import copy
import torch
from tensorboardX import SummaryWriter

from ding.config import compile_config
from ding.worker import BaseLearner, Episode1v1Collector, OnevOneEvaluator, NaiveReplayBuffer
from ding.envs import BaseEnvManager, DingEnvWrapper
from ding.policy import PPOPolicy
from dizoo.overcooked.models.overcooked_vac import BaselineVAC
from ding.utils import set_pkg_seed
from dizoo.overcooked.envs import OvercookGameEnv
from dizoo.overcooked.config import overcooked_demo_ppo_config


def wrapped_overcookgame():
    return OvercookGameEnv({})


def main(cfg, seed=0, max_iterations=int(1e10)):
    cfg.exp_name = 'selfplay_demo_ppo'
    cfg = compile_config(
        cfg,
        BaseEnvManager,
        PPOPolicy,
        BaseLearner,
        Episode1v1Collector,
        OnevOneEvaluator,
        NaiveReplayBuffer,
        save_cfg=True
    )
    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    collector_env = BaseEnvManager(env_fn=[wrapped_overcookgame for _ in range(collector_env_num)], cfg=cfg.env.manager)
    evaluator_env1 = BaseEnvManager(
        env_fn=[wrapped_overcookgame for _ in range(evaluator_env_num)], cfg=cfg.env.manager
    )
    evaluator_env2 = BaseEnvManager(
        env_fn=[wrapped_overcookgame for _ in range(evaluator_env_num)], cfg=cfg.env.manager
    )

    collector_env.seed(seed)
    evaluator_env1.seed(seed, dynamic_seed=False)
    evaluator_env2.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

    model1 = BaselineVAC(**cfg.policy.model)
    policy1 = PPOPolicy(cfg.policy, model=model1)
    model2 = BaselineVAC(**cfg.policy.model)
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
    # collect_mode ppo use multimonial sample for selecting action
    evaluator1_cfg = copy.deepcopy(cfg.policy.eval.evaluator)
    evaluator1_cfg.stop_value = cfg.env.stop_value
    evaluator1 = OnevOneEvaluator(
        evaluator1_cfg,
        evaluator_env1, [policy1.collect_mode, policy2.collect_mode],
        tb_logger,
        exp_name=cfg.exp_name,
        instance_name='selfplay_evaluator1'
    )
    evaluator2_cfg = copy.deepcopy(cfg.policy.eval.evaluator)
    evaluator2_cfg.stop_value = cfg.env.stop_value
    evaluator2 = OnevOneEvaluator(
        evaluator2_cfg,
        evaluator_env2, [policy2.collect_mode, policy1.collect_mode],
        tb_logger,
        exp_name=cfg.exp_name,
        instance_name='selfplay_evaluator2'
    )

    for _ in range(max_iterations):
        if evaluator1.should_eval(learner1.train_iter):
            stop_flag1, reward = evaluator1.eval(learner1.save_checkpoint, learner1.train_iter, collector.envstep)
            tb_logger.add_scalar('selfplay1_evaluator_step/reward_mean', reward, collector.envstep)
        if evaluator2.should_eval(learner1.train_iter):
            stop_flag2, reward = evaluator2.eval(learner1.save_checkpoint, learner1.train_iter, collector.envstep)
            tb_logger.add_scalar('selfplay2_evaluator_step/reward_mean', reward, collector.envstep)
        if stop_flag1 and stop_flag2:
            break
        train_data, _ = collector.collect(train_iter=learner1.train_iter)
        for i in range(cfg.policy.learn.update_per_collect):
            learner1.train(train_data[0], collector.envstep)
            learner2.train(train_data[1], collector.envstep)


if __name__ == "__main__":
    main(overcooked_demo_ppo_config)
