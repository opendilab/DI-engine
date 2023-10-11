import os
import gym
import numpy as np
from tensorboardX import SummaryWriter
import torch
from rocket_recycling.rocket import Rocket

from ditk import logging
from ding.model import VAC
from ding.policy import PPOPolicy
from ding.envs import DingEnvWrapper, BaseEnvManagerV2, EvalEpisodeReturnWrapper
from ding.config import compile_config
from ding.framework import task
from ding.framework.context import OnlineRLContext
from ding.framework.middleware import multistep_trainer, StepCollector, interaction_evaluator, CkptSaver, \
    gae_estimator, termination_checker
from ding.utils import set_pkg_seed
from dizoo.rocket.config.rocket_hover_ppo_config import main_config, create_config


class RocketHoverWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self._observation_space = gym.spaces.Box(low=float("-inf"), high=float("inf"), shape=(8, ), dtype=np.float32)
        self._action_space = gym.spaces.Discrete(9)
        self._action_space.seed(0)  # default seed
        self.reward_range = (float('-inf'), float('inf'))


def wrapped_rocket_env(task, max_steps):
    return DingEnvWrapper(
        Rocket(task=task, max_steps=max_steps),
        cfg={'env_wrapper': [
            lambda env: RocketHoverWrapper(env),
            lambda env: EvalEpisodeReturnWrapper(env),
        ]}
    )


def main():
    logging.getLogger().setLevel(logging.INFO)
    main_config.policy.cuda = True
    print('torch.cuda.is_available(): ', torch.cuda.is_available())
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)
    num_seed = 3
    for seed_i in range(num_seed):
        main_config.exp_name = f'task_rocket_hovering_onppo_seed{seed_i}'
        tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'seed' + str(seed_i)))
        with task.start(async_mode=False, ctx=OnlineRLContext()):
            collector_env = BaseEnvManagerV2(
                env_fn=[
                    lambda: wrapped_rocket_env(cfg.env.task, cfg.env.max_steps)
                    for _ in range(cfg.env.collector_env_num)
                ],
                cfg=cfg.env.manager
            )
            evaluator_env = BaseEnvManagerV2(
                env_fn=[
                    lambda: wrapped_rocket_env(cfg.env.task, cfg.env.max_steps)
                    for _ in range(cfg.env.evaluator_env_num)
                ],
                cfg=cfg.env.manager
            )

            # evaluator_env.enable_save_replay()

            set_pkg_seed(seed_i, use_cuda=cfg.policy.cuda)

            model = VAC(**cfg.policy.model)
            policy = PPOPolicy(cfg.policy, model=model)

            def _add_scalar(ctx):
                if ctx.eval_value != -np.inf:
                    tb_logger.add_scalar('evaluator_step/reward', ctx.eval_value, global_step=ctx.env_step)
                    collector_rewards = [ctx.trajectories[i]['reward'] for i in range(len(ctx.trajectories))]
                    collector_mean_reward = sum(collector_rewards) / len(ctx.trajectories)
                    collector_max_reward = max(collector_rewards)
                    collector_min_reward = min(collector_rewards)
                    tb_logger.add_scalar('collecter_step/mean_reward', collector_mean_reward, global_step=ctx.env_step)
                    tb_logger.add_scalar('collecter_step/max_reward', collector_max_reward, global_step=ctx.env_step)
                    tb_logger.add_scalar('collecter_step/min_reward', collector_min_reward, global_step=ctx.env_step)

            task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
            task.use(StepCollector(cfg, policy.collect_mode, collector_env))
            task.use(_add_scalar)
            task.use(gae_estimator(cfg, policy.collect_mode))
            task.use(multistep_trainer(cfg, policy.learn_mode))
            task.use(CkptSaver(policy, cfg.exp_name, train_freq=100))
            task.use(termination_checker(max_env_step=int(10e7)))
            task.run()


if __name__ == "__main__":
    main()
