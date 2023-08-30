from easydict import EasyDict
import ding.envs.gym_env

cfg = dict(
    exp_name='HalfCheetah-v3-A2C',
    env=dict(
        env_id='HalfCheetah-v3',
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        collector_env_num=1,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=12000,
        env_wrapper='mujoco_default',
        act_scale=True,
        rew_clip=True,
    ),
    policy=dict(
        cuda=True,
        action_space='continuous',
        model=dict(
            action_space='continuous',
            obs_shape=17,
            action_shape=6,
        ),
        learn=dict(
            batch_size=256,
            learning_rate=0.0003,
            value_weight=0.5,
            entropy_weight=0.01,
            grad_norm=0.5,
            ignore_done=True,
            adv_norm=True,
        ),
        collect=dict(
            n_sample=256,
            discount_factor=0.99,
            gae_lambda=0.95,
        ),
    ),
    wandb_logger=dict(
        gradient_logger=True, video_logger=True, plot_logger=True, action_logger=True, return_logger=False
    ),
)

cfg = EasyDict(cfg)

env = ding.envs.gym_env.env
