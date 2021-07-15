import os
import gym
import numpy as np
import torch
from tensorboardX import SummaryWriter

from ding.config import compile_config
from ding.worker import BaseLearner, Episode1v1Collector, OnevOneEvaluator, NaiveReplayBuffer
from ding.envs import BaseEnvManager, DingEnvWrapper
from ding.policy import PPOPolicy
from ding.model import VAC
from ding.utils import set_pkg_seed
from game_env import GameEnv
from league_demo_ppo_config import league_demo_ppo_config


class EvalPolicy1:

    def forward(self, data: dict) -> dict:
        return {env_id: {'action': torch.zeros(1)} for env_id in data.keys()}

    def reset(self, data_id: list = []) -> None:
        pass


class EvalPolicy2:

    def forward(self, data: dict) -> dict:
        return {
            env_id: {
                'action': torch.from_numpy(np.random.choice([0, 1], p=[0.5, 0.5], size=(1, )))
            }
            for env_id in data.keys()
        }

    def reset(self, data_id: list = []) -> None:
        pass


def main(cfg, seed=0, max_iterations=int(1e10)):
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
    collector_env = BaseEnvManager(env_fn=[GameEnv for _ in range(collector_env_num)], cfg=cfg.env.manager)
    evaluator_env1 = BaseEnvManager(env_fn=[GameEnv for _ in range(evaluator_env_num)], cfg=cfg.env.manager)
    evaluator_env2 = BaseEnvManager(env_fn=[GameEnv for _ in range(evaluator_env_num)], cfg=cfg.env.manager)

    collector_env.seed(seed)
    evaluator_env1.seed(seed, dynamic_seed=False)
    evaluator_env2.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

    model = VAC(**cfg.policy.model)
    policy = PPOPolicy(cfg.policy, model=model)
    eval_policy1 = EvalPolicy1()
    eval_policy2 = EvalPolicy2()

    tb_logger = SummaryWriter(os.path.join('./log/', 'serial'))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger)
    collector = Episode1v1Collector(
        cfg.policy.collect.collector, collector_env, [policy.collect_mode, policy.collect_mode], tb_logger
    )
    # collect_mode ppo use multimonial sample for selecting action
    evaluator1 = OnevOneEvaluator(
        cfg.policy.eval.evaluator, evaluator_env1, [policy.collect_mode, eval_policy1], tb_logger, name='fixed'
    )
    evaluator2 = OnevOneEvaluator(
        cfg.policy.eval.evaluator, evaluator_env2, [policy.collect_mode, eval_policy2], tb_logger, name='uniform'
    )

    for _ in range(max_iterations):
        if evaluator1.should_eval(learner.train_iter):
            stop_flag1, reward = evaluator1.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            tb_logger.add_scalar('evaluator1_step/reward_mean', reward, collector.envstep)
        if evaluator2.should_eval(learner.train_iter):
            stop_flag2, reward = evaluator2.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            tb_logger.add_scalar('evaluator2_step/reward_mean', reward, collector.envstep)
        if stop_flag1 and stop_flag2:
            break
        train_data = collector.collect(train_iter=learner.train_iter)
        for d in train_data:
            d['adv'] = d['reward']
        for i in range(cfg.policy.learn.update_per_collect):
            learner.train(train_data, collector.envstep)


if __name__ == "__main__":
    main(league_demo_ppo_config)
