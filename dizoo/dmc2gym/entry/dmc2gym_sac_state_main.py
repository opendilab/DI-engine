from ditk import logging
from ding.model import QAC
from ding.policy import SACPolicy
from ding.envs import BaseEnvManagerV2
from ding.data import DequeBuffer
from ding.config import compile_config
from ding.framework import task
from ding.framework.context import OnlineRLContext
from ding.framework.middleware import data_pusher, StepCollector, interaction_evaluator, \
    CkptSaver, OffPolicyLearner, termination_checker
from ding.utils import set_pkg_seed
from dizoo.dmc2gym.envs.dmc2gym_env import DMC2GymEnv
from dizoo.dmc2gym.config.dmc2gym_sac_state_config import main_config, create_config
import numpy as np
from tensorboardX import SummaryWriter
import os

def main():
    logging.getLogger().setLevel(logging.INFO)
    main_config.exp_name = 'dmc2gym_sac_state_seed0_817'
    main_config.policy.cuda = True
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)

    num_seed = 1
    for seed_i in range(num_seed):
        tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'seed'+str(seed_i)))
        
        with task.start(async_mode=False, ctx=OnlineRLContext()):
            collector_env = BaseEnvManagerV2(
                env_fn=[lambda: DMC2GymEnv(cfg.env) for _ in range(cfg.env.collector_env_num)], cfg=cfg.env.manager
            )
            evaluator_env = BaseEnvManagerV2(
                env_fn=[lambda: DMC2GymEnv(cfg.env) for _ in range(cfg.env.evaluator_env_num)], cfg=cfg.env.manager
            )

            set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

            model = QAC(**cfg.policy.model)
            buffer_ = DequeBuffer(size=cfg.policy.other.replay_buffer.replay_buffer_size)
            policy = SACPolicy(cfg.policy, model=model)

            def _add_scalar(ctx):
                if ctx.eval_value != -np.inf:
                    tb_logger.add_scalar('evaluator_step/reward', ctx.eval_value, global_step= ctx.env_step)
                    collector_rewards = [ctx.trajectories[i]['reward'] for i in range(len(ctx.trajectories))]
                    collector_mean_reward = sum(collector_rewards) / len(ctx.trajectories)
                    collector_max_reward = max(collector_rewards)
                    collector_min_reward = min(collector_rewards)
                    tb_logger.add_scalar('collecter_step/mean_reward', collector_mean_reward, global_step= ctx.env_step)
                    tb_logger.add_scalar('collecter_step/max_reward', collector_max_reward, global_step= ctx.env_step)
                    tb_logger.add_scalar('collecter_step/min_reward', collector_min_reward, global_step= ctx.env_step)

            task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
            task.use(
                StepCollector(cfg, policy.collect_mode, collector_env, random_collect_size=cfg.policy.random_collect_size)
            )
            task.use(_add_scalar)
            task.use(data_pusher(cfg, buffer_))
            task.use(OffPolicyLearner(cfg, policy.learn_mode, buffer_))
            task.use(CkptSaver(cfg, policy, train_freq=100))
            task.use(termination_checker(max_env_step=int(10e7)))
            task.run()


if __name__ == "__main__":
    main()
