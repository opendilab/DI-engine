import os
import pathlib
from ditk import logging
from ding.model.template.qac import QAC
from ding.policy import TD3Policy
from ding.envs import DingEnvWrapper, BaseEnvManagerV2, SubprocessEnvManagerV2
from ding.data import DequeBuffer
from ding.config import compile_config
from ding.framework import task
from ding.framework.context import OnlineRLContext
from ding.framework.middleware import OffPolicyLearner, StepCollector, interaction_evaluator, data_pusher, CkptSaver, termination_checker, wandb_online_logger
from ding.utils import set_pkg_seed
from ding.utils.log_helper import build_logger
from dizoo.mujoco.envs.mujoco_env import MujocoEnv
from easydict import EasyDict
import wandb

hopper_td3_config = dict(
    exp_name='hopper_td3_wandb_seed0',
    seed=0,
    env=dict(
        env_id='Hopper-v3',
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=6000,
    ),
    policy=dict(
        cuda=True,
        random_collect_size=25000,
        model=dict(
            obs_shape=11,
            action_shape=3,
            twin_critic=True,
            actor_head_hidden_size=256,
            critic_head_hidden_size=256,
            action_space='regression',
        ),
        logger=dict(record_path='./video_hopper_td3', gradient_logger=True, plot_logger=True, action_logger=None),
        learn=dict(
            update_per_collect=1,
            batch_size=256,
            learning_rate_actor=1e-3,
            learning_rate_critic=1e-3,
            ignore_done=False,
            target_theta=0.005,
            discount_factor=0.99,
            actor_update_freq=2,
            noise=True,
            noise_sigma=0.2,
            noise_range=dict(
                min=-0.5,
                max=0.5,
            ),
        ),
        collect=dict(
            n_sample=1,
            unroll_len=1,
            noise_sigma=0.1,
        ),
        other=dict(replay_buffer=dict(replay_buffer_size=1000000, ), ),
    )
)

hopper_td3_config = EasyDict(hopper_td3_config)
main_config = hopper_td3_config

hopper_td3_create_config = dict(
    env=dict(
        type='mujoco',
        import_names=['dizoo.mujoco.envs.mujoco_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='td3',
        import_names=['ding.policy.td3'],
    ),
    replay_buffer=dict(type='naive', ),
)
hopper_td3_create_config = EasyDict(hopper_td3_create_config)
create_config = hopper_td3_create_config


def main(seed=0, max_env_step=10000000):
    logging.getLogger().setLevel(logging.INFO)
    cfg = compile_config(main_config, create_cfg=create_config, auto=True)
    cfg.env.seed = seed

    wandb.init(
        # Set the project where this run will be logged
        project='hopper-td3-0111',
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=str(main_config["DI-toolkit-hpo-id"]),
        # Track hyperparameters and run metadata
        config=cfg
    )

    # logger_, tb_logger = build_logger(path='./log/hopper_td3/seed' + str(seed),
    #                                   need_tb=True)
    with task.start(async_mode=False, ctx=OnlineRLContext()):
        collector_env = SubprocessEnvManagerV2(
            env_fn=[lambda: MujocoEnv(cfg.env) for _ in range(cfg.env.collector_env_num)], cfg=cfg.env.manager
        )
        evaluator_env = SubprocessEnvManagerV2(
            env_fn=[lambda: MujocoEnv(cfg.env) for _ in range(cfg.env.evaluator_env_num)], cfg=cfg.env.manager
        )
        cfg.policy.logger.record_path = './' + cfg.exp_name + '/video'
        evaluator_env.enable_save_replay(replay_path=cfg.policy.logger.record_path)

        set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)

        model = QAC(**cfg.policy.model)
        buffer_ = DequeBuffer(size=cfg.policy.other.replay_buffer.replay_buffer_size)
        policy = TD3Policy(cfg.policy, model)

        task.use(interaction_evaluator(cfg, policy.eval_mode, evaluator_env))
        task.use(
            StepCollector(cfg, policy.collect_mode, collector_env, random_collect_size=cfg.policy.random_collect_size)
        )
        task.use(data_pusher(cfg, buffer_))
        task.use(OffPolicyLearner(cfg, policy.learn_mode, buffer_))
        task.use(CkptSaver(policy=policy,save_dir=os.path.join(cfg["exp_name"],"model"), train_freq=100))
        task.use(wandb_online_logger(cfg.policy.logger, evaluator_env, model))
        task.use(termination_checker(max_env_step=max_env_step))
        task.run()


if __name__ == "__main__":
    main(seed=main_config.seed, max_env_step=10000000) 
