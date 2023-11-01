import datetime
import torch
try:
    torch.multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass
from easydict import EasyDict
from ditk import logging
from ding.model import DQN
from ding.policy import DQNFastPolicy
from ding.envs.env_manager.envpool_env_manager import PoolEnvManagerV2
from ding.data import DequeBuffer
from ding.config import compile_config
from ding.framework import task, ding_init
from ding.framework.context import OnlineRLContext
from ding.framework.middleware import envpool_evaluator, data_pusher, \
    eps_greedy_handler, CkptSaver, ContextExchanger, ModelExchanger, online_logger, \
    termination_checker, wandb_online_logger, epoch_timer, EnvpoolStepCollector, EnvpoolOffPolicyLearner
from ding.utils import set_pkg_seed

from dizoo.atari.config.serial import pong_dqn_envpool_config


def main(cfg):
    logging.getLogger().setLevel(logging.INFO)
    cfg.exp_name = 'Pong-v5-DQN-envpool-' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    collector_env_cfg = EasyDict(
        {
            'env_id': cfg.env.env_id,
            'env_num': cfg.env.collector_env_num,
            'batch_size': cfg.env.collector_batch_size,
            # env wrappers
            'episodic_life': True,  # collector: True
            'reward_clip': False,  # collector: True
            'gray_scale': cfg.env.get('gray_scale', True),
            'stack_num': cfg.env.get('stack_num', 4),
        }
    )
    cfg.env["collector_env_cfg"] = collector_env_cfg
    evaluator_env_cfg = EasyDict(
        {
            'env_id': cfg.env.env_id,
            'env_num': cfg.env.evaluator_env_num,
            'batch_size': cfg.env.evaluator_batch_size,
            # env wrappers
            'episodic_life': False,  # evaluator: False
            'reward_clip': False,  # evaluator: False
            'gray_scale': cfg.env.get('gray_scale', True),
            'stack_num': cfg.env.get('stack_num', 4),
        }
    )
    cfg.env["evaluator_env_cfg"] = evaluator_env_cfg
    cfg = compile_config(cfg, PoolEnvManagerV2, DQNFastPolicy, save_cfg=task.router.node_id == 0)
    ding_init(cfg)
    with task.start(async_mode=False, ctx=OnlineRLContext()):
        collector_env = PoolEnvManagerV2(cfg.env.collector_env_cfg)
        evaluator_env = PoolEnvManagerV2(cfg.env.evaluator_env_cfg)
        collector_env.seed(cfg.seed)
        evaluator_env.seed(cfg.seed)
        set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

        model = DQN(**cfg.policy.model)
        buffer_ = DequeBuffer(size=cfg.policy.other.replay_buffer.replay_buffer_size)
        policy = DQNFastPolicy(cfg.policy, model=model)

        # Consider the case with multiple processes
        if task.router.is_active:
            # You can use labels to distinguish between workers with different roles,
            # here we use node_id to distinguish.
            if task.router.node_id == 0:
                task.add_role(task.role.LEARNER)
            elif task.router.node_id == 1:
                task.add_role(task.role.EVALUATOR)
            else:
                task.add_role(task.role.COLLECTOR)

            # Sync their context and model between each worker.
            task.use(ContextExchanger(skip_n_iter=1))
            task.use(ModelExchanger(model))

        task.use(epoch_timer())

        # Here is the part of single process pipeline.
        task.use(envpool_evaluator(cfg, policy.eval_mode, evaluator_env))
        task.use(eps_greedy_handler(cfg))
        task.use(
            EnvpoolStepCollector(
                cfg,
                policy.collect_mode,
                collector_env,
                random_collect_size=cfg.policy.random_collect_size \
                       if hasattr(cfg.policy, 'random_collect_size') else 0,
                    )
                )
        task.use(data_pusher(cfg, buffer_))
        task.use(EnvpoolOffPolicyLearner(cfg, policy, buffer_))
        task.use(online_logger(train_show_freq=10))
        task.use(
            wandb_online_logger(
                metric_list=policy._monitor_vars_learn(),
                model=policy._model,
                exp_config=cfg,
                anonymous=True,
                project_name=cfg.exp_name,
                wandb_sweep=False,
            )
        )

        #task.use(CkptSaver(policy, cfg.exp_name, train_freq=1000))
        task.use(termination_checker(max_env_step=100))

        task.run()


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--collector_env_num", type=int, default=8, help="collector env number")
    parser.add_argument("--collector_batch_size", type=int, default=8, help="collector batch size")
    arg = parser.parse_args()

    pong_dqn_envpool_config.env.collector_env_num = arg.collector_env_num
    pong_dqn_envpool_config.env.collector_batch_size = arg.collector_batch_size
    pong_dqn_envpool_config.seed = arg.seed
    pong_dqn_envpool_config.env.stop_value = 2000
    pong_dqn_envpool_config.nstep = 3
    pong_dqn_envpool_config.policy.nstep = 3
    pong_dqn_envpool_config.seed = arg.seed

    pong_dqn_envpool_config.policy.learn.update_per_collect = 2
    pong_dqn_envpool_config.policy.learn.batch_size = 32
    pong_dqn_envpool_config.policy.learn.learning_rate = 0.0001
    pong_dqn_envpool_config.policy.learn.target_update_freq = 0
    pong_dqn_envpool_config.policy.learn.target_update = 0.04

    main(pong_dqn_envpool_config)
