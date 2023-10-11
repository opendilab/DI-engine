from ditk import logging
from ding.model import ContinuousQAC
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
    main_config.exp_name = 'dmc2gym_sac_state_nseed_5M'
    main_config.policy.cuda = True
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)

    num_seed = 4
    for seed_i in range(num_seed):
        tb_logger = SummaryWriter(os.path.join('./{}/log/'.format(cfg.exp_name), 'seed' + str(seed_i)))

        with task.start(async_mode=False, ctx=OnlineRLContext()):
            collector_env = BaseEnvManagerV2(
                env_fn=[lambda: DMC2GymEnv(cfg.env) for _ in range(cfg.env.collector_env_num)], cfg=cfg.env.manager
            )
            evaluator_env = BaseEnvManagerV2(
                env_fn=[lambda: DMC2GymEnv(cfg.env) for _ in range(cfg.env.evaluator_env_num)], cfg=cfg.env.manager
            )

            set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

            model = ContinuousQAC(**cfg.policy.model)
            buffer_ = DequeBuffer(size=cfg.policy.other.replay_buffer.replay_buffer_size)
            policy = SACPolicy(cfg.policy, model=model)

            def _add_scalar(ctx):
                if ctx.eval_value != -np.inf:
                    tb_logger.add_scalar('evaluator_step/reward', ctx.eval_value, global_step=ctx.env_step)
                    collector_rewards = [ctx.trajectories[i]['reward'] for i in range(len(ctx.trajectories))]
                    collector_mean_reward = sum(collector_rewards) / len(ctx.trajectories)
                    # collector_max_reward = max(collector_rewards)
                    # collector_min_reward = min(collector_rewards)
                    tb_logger.add_scalar('collecter_step/mean_reward', collector_mean_reward, global_step=ctx.env_step)
                    # tb_logger.add_scalar('collecter_step/max_reward', collector_max_reward, global_step= ctx.env_step)
                    # tb_logger.add_scalar('collecter_step/min_reward', collector_min_reward, global_step= ctx.env_step)
                    tb_logger.add_scalar(
                        'collecter_step/avg_env_step_per_episode',
                        ctx.env_step / ctx.env_episode,
                        global_step=ctx.env_step
                    )

            def _add_train_scalar(ctx):
                len_train = len(ctx.train_output)
                cur_lr_q_avg = sum([ctx.train_output[i]['cur_lr_q'] for i in range(len_train)]) / len_train
                cur_lr_p_avg = sum([ctx.train_output[i]['cur_lr_p'] for i in range(len_train)]) / len_train
                critic_loss_avg = sum([ctx.train_output[i]['critic_loss'] for i in range(len_train)]) / len_train
                policy_loss_avg = sum([ctx.train_output[i]['policy_loss'] for i in range(len_train)]) / len_train
                total_loss_avg = sum([ctx.train_output[i]['total_loss'] for i in range(len_train)]) / len_train
                tb_logger.add_scalar('learner_step/cur_lr_q_avg', cur_lr_q_avg, global_step=ctx.env_step)
                tb_logger.add_scalar('learner_step/cur_lr_p_avg', cur_lr_p_avg, global_step=ctx.env_step)
                tb_logger.add_scalar('learner_step/critic_loss_avg', critic_loss_avg, global_step=ctx.env_step)
                tb_logger.add_scalar('learner_step/policy_loss_avg', policy_loss_avg, global_step=ctx.env_step)
                tb_logger.add_scalar('learner_step/total_loss_avg', total_loss_avg, global_step=ctx.env_step)

            task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
            task.use(
                StepCollector(
                    cfg, policy.collect_mode, collector_env, random_collect_size=cfg.policy.random_collect_size
                )
            )
            task.use(_add_scalar)
            task.use(data_pusher(cfg, buffer_))
            task.use(OffPolicyLearner(cfg, policy.learn_mode, buffer_))
            task.use(_add_train_scalar)
            task.use(CkptSaver(policy, cfg.exp_name, train_freq=int(1e5)))
            task.use(termination_checker(max_env_step=int(5e6)))
            task.run()


if __name__ == "__main__":
    main()
