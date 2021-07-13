import os
import gym
from tensorboardX import SummaryWriter

from ding.config import compile_config
from ding.worker import BaseLearner, Episode1v1Collector, BaseSerialEvaluator, NaiveReplayBuffer
from ding.envs import BaseEnvManager, DingEnvWrapper
from ding.policy import PPOPolicy
from ding.model import VAC
from ding.utils import set_pkg_seed
from game_env import GameEnv
from league_demo_ppo_config import league_demo_ppo_config


def main(cfg, seed=0, max_iterations=int(1e10)):
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
    collector_env = BaseEnvManager(env_fn=[GameEnv for _ in range(collector_env_num)], cfg=cfg.env.manager)
    evaluator_env = BaseEnvManager(env_fn=[GameEnv for _ in range(evaluator_env_num)], cfg=cfg.env.manager)

    collector_env.seed(seed)
    evaluator_env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

    model = VAC(**cfg.policy.model)
    policy = PPOPolicy(cfg.policy, model=model)
    tb_logger = SummaryWriter(os.path.join('./log/', 'serial'))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger)
    collector = Episode1v1Collector(
        cfg.policy.collect.collector, collector_env, [policy.collect_mode, policy.collect_mode], tb_logger
    )
    evaluator = BaseSerialEvaluator(cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger)

    for _ in range(max_iterations):
        if False and evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break
        train_data = collector.collect(train_iter=learner.train_iter)
        for i in range(cfg.policy.learn.update_per_collect):
            learner.train(train_data, collector.envstep)


if __name__ == "__main__":
    main(league_demo_ppo_config)
