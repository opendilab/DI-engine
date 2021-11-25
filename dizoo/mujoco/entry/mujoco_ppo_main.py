import os
import gym
from tensorboardX import SummaryWriter
from easydict import EasyDict

from ding.config import compile_config
from ding.worker import BaseLearner, SampleSerialCollector, InteractionSerialEvaluator, NaiveReplayBuffer
from ding.envs import BaseEnvManager, DingEnvWrapper
from ding.policy import PPOPolicy
from ding.model import VAC
from ding.utils import set_pkg_seed
from dizoo.classic_control.pendulum.envs import PendulumEnv
from dizoo.mujoco.envs.mujoco_env import MujocoEnv
from dizoo.classic_control.pendulum.config.pendulum_ppo_config import pendulum_ppo_config
from dizoo.mujoco.config.hopper_ppo_default_config import hopper_ppo_default_config


def main(cfg, seed=0, max_iterations=int(1e10)):
    cfg = compile_config(
        cfg,
        BaseEnvManager,
        PPOPolicy,
        BaseLearner,
        SampleSerialCollector,
        InteractionSerialEvaluator,
        NaiveReplayBuffer,
        save_cfg=True
    )
    collector_env_num, evaluator_env_num = cfg.env.collector_env_num, cfg.env.evaluator_env_num
    collector_env = BaseEnvManager(
        env_fn=[lambda: MujocoEnv(cfg.env) for _ in range(collector_env_num)], cfg=cfg.env.manager
    )
    evaluator_env = BaseEnvManager(
        env_fn=[lambda: MujocoEnv(cfg.env) for _ in range(evaluator_env_num)], cfg=cfg.env.manager
    )

    collector_env.seed(seed, dynamic_seed=True)
    evaluator_env.seed(seed, dynamic_seed=False)
    set_pkg_seed(seed, use_cuda=cfg.policy.cuda)

    model = VAC(**cfg.policy.model)
    policy = PPOPolicy(cfg.policy, model=model)
    tb_logger = SummaryWriter(os.path.join('./log/', 'serial'))
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger)
    collector = SampleSerialCollector(cfg.policy.collect.collector, collector_env, policy.collect_mode, tb_logger)
    evaluator = InteractionSerialEvaluator(cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger)

    for _ in range(max_iterations):
        if evaluator.should_eval(learner.train_iter):
            stop, reward = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
            if stop:
                break
        new_data = collector.collect(train_iter=learner.train_iter)
        learner.train(new_data, collector.envstep)


if __name__ == "__main__":
    main(hopper_ppo_default_config)
