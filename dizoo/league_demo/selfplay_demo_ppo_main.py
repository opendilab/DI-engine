import os
import gym
import numpy as np
import copy
import torch
from tensorboardX import SummaryWriter

from ding.config import compile_config
from ding.worker import BaseLearner, BattleInteractionSerialEvaluator, NaiveReplayBuffer
from ding.envs import BaseEnvManager, DingEnvWrapper
from ding.policy import PPOPolicy
from ding.model import VAC
from ding.utils import set_pkg_seed
from dizoo.league_demo.game_env import GameEnv
from dizoo.league_demo.league_demo_collector import LeagueDemoCollector
from dizoo.league_demo.league_demo_ppo_config import league_demo_ppo_config


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


def main(cfg, seed=0, max_train_iter=int(1e8), max_env_step=int(1e8)):
    cfg.exp_name = 'selfplay_demo_ppo'
    cfg = compile_config(
        cfg,
        BaseEnvManager,
        PPOPolicy,
        BaseLearner,
        LeagueDemoCollector,
        BattleInteractionSerialEvaluator,
        NaiveReplayBuffer,
        save_cfg=True
    )
    env_type = cfg.env.env_type
    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    collector_env = BaseEnvManager(
        env_fn=[lambda: GameEnv(env_type) for _ in range(collector_env_num)], cfg=cfg.env.manager
    )
    evaluator_env1 = BaseEnvManager(
        env_fn=[lambda: GameEnv(env_type) for _ in range(evaluator_env_num)], cfg=cfg.env.manager
    )
    evaluator_env2 = BaseEnvManager(
        env_fn=[lambda: GameEnv(env_type) for _ in range(evaluator_env_num)], cfg=cfg.env.manager
    )

    collector_env.seed(seed)
    evaluator_env1.seed(seed, dynamic_seed=False)
    evaluator_env2.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

    model1 = VAC(**cfg.policy.model)
    policy1 = PPOPolicy(cfg.policy, model=model1)
    model2 = VAC(**cfg.policy.model)
    policy2 = PPOPolicy(cfg.policy, model=model2)
    eval_policy1 = EvalPolicy1()
    eval_policy2 = EvalPolicy2()

    tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'serial'))
    learner1 = BaseLearner(
        cfg.policy.learn.learner, policy1.learn_mode, tb_logger, exp_name=cfg.exp_name, instance_name='learner1'
    )
    learner2 = BaseLearner(
        cfg.policy.learn.learner, policy2.learn_mode, tb_logger, exp_name=cfg.exp_name, instance_name='learner2'
    )
    collector = LeagueDemoCollector(
        cfg.policy.collect.collector,
        collector_env, [policy1.collect_mode, policy2.collect_mode],
        tb_logger,
        exp_name=cfg.exp_name
    )
    # collect_mode ppo use multinomial sample for selecting action
    evaluator1_cfg = copy.deepcopy(cfg.policy.eval.evaluator)
    evaluator1_cfg.stop_value = cfg.env.stop_value[0]
    evaluator1 = BattleInteractionSerialEvaluator(
        evaluator1_cfg,
        evaluator_env1, [policy1.collect_mode, eval_policy1],
        tb_logger,
        exp_name=cfg.exp_name,
        instance_name='fixed_evaluator'
    )
    evaluator2_cfg = copy.deepcopy(cfg.policy.eval.evaluator)
    evaluator2_cfg.stop_value = cfg.env.stop_value[1]
    evaluator2 = BattleInteractionSerialEvaluator(
        evaluator2_cfg,
        evaluator_env2, [policy1.collect_mode, eval_policy2],
        tb_logger,
        exp_name=cfg.exp_name,
        instance_name='uniform_evaluator'
    )

    while True:
        if evaluator1.should_eval(learner1.train_iter):
            stop_flag1, _ = evaluator1.eval(learner1.save_checkpoint, learner1.train_iter, collector.envstep)
        if evaluator2.should_eval(learner1.train_iter):
            stop_flag2, _ = evaluator2.eval(learner1.save_checkpoint, learner1.train_iter, collector.envstep)
        if stop_flag1 and stop_flag2:
            break
        train_data, _ = collector.collect(train_iter=learner1.train_iter)
        for data in train_data:
            for d in data:
                d['adv'] = d['reward']
        for i in range(cfg.policy.learn.update_per_collect):
            learner1.train(train_data[0], collector.envstep)
            learner2.train(train_data[1], collector.envstep)
        if collector.envstep >= max_env_step or learner1.train_iter >= max_train_iter:
            break


if __name__ == "__main__":
    main(league_demo_ppo_config)
