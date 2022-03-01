import os
import gym
import numpy as np
import copy
import torch
from tensorboardX import SummaryWriter
from functools import partial

from ding.config import compile_config
from ding.worker import BaseLearner, BattleSampleSerialCollector, NaiveReplayBuffer, InteractionSerialEvaluator
from ding.envs import SyncSubprocessEnvManager
from ding.policy import PPOPolicy
from ding.model import VAC
from ding.utils import set_pkg_seed
from dizoo.slime_volley.envs import SlimeVolleyEnv
from dizoo.slime_volley.config.slime_volley_ppo_config import main_config


def main(cfg, seed=0, max_iterations=int(1e10)):
    """
    Overview:
        Naive self-play, no any historial player.
    """
    cfg = compile_config(
        cfg,
        SyncSubprocessEnvManager,
        PPOPolicy,
        BaseLearner,
        BattleSampleSerialCollector,
        InteractionSerialEvaluator,
        NaiveReplayBuffer,
        save_cfg=True
    )
    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    collector_env_cfg = copy.deepcopy(cfg.env)
    collector_env_cfg.agent_vs_agent = True
    evaluator_env_cfg = copy.deepcopy(cfg.env)
    evaluator_env_cfg.agent_vs_agent = False
    collector_env = SyncSubprocessEnvManager(
        env_fn=[partial(SlimeVolleyEnv, collector_env_cfg) for _ in range(collector_env_num)], cfg=cfg.env.manager
    )
    evaluator_env = SyncSubprocessEnvManager(
        env_fn=[partial(SlimeVolleyEnv, evaluator_env_cfg) for _ in range(evaluator_env_num)], cfg=cfg.env.manager
    )

    collector_env.seed(seed)
    evaluator_env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

    model = VAC(**cfg.policy.model)
    policy = PPOPolicy(cfg.policy, model=model)

    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    learner = BaseLearner(
        cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name, instance_name='learner1'
    )
    collector = BattleSampleSerialCollector(
        cfg.policy.collect.collector,
        collector_env, [policy.collect_mode, policy.collect_mode],
        tb_logger,
        exp_name=cfg.exp_name
    )
    evaluator_cfg = copy.deepcopy(cfg.policy.eval.evaluator)
    evaluator_cfg.stop_value = cfg.env.stop_value
    evaluator = InteractionSerialEvaluator(
        evaluator_cfg,
        evaluator_env,
        policy.eval_mode,
        tb_logger,
        exp_name=cfg.exp_name,
        instance_name='builtin_ai_evaluator'
    )

    learner.call_hook('before_run')
    for _ in range(max_iterations):
        if evaluator.should_eval(learner.train_iter):
            stop_flag, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop_flag:
                break
        new_data, _ = collector.collect(train_iter=learner.train_iter)
        train_data = new_data[0] + new_data[1]
        learner.train(train_data, collector.envstep)
    learner.call_hook('after_run')


if __name__ == "__main__":
    main(main_config)
