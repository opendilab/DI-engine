from easydict import EasyDict

cfg = dict(
    exp_name='Hopper-v3-TD3',
    seed=0,
    env=dict(
        env_id='Hopper-v3',
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=6000,
        env_wrapper='mujoco_default',
        act_scale=True,
        rew_clip=True,
    ),
    policy=dict(
        cuda=True,
        random_collect_size=25000,
        model=dict(
            obs_shape=11,
            action_shape=3,
            actor_head_hidden_size=256,
            critic_head_hidden_size=256,
            action_space='regression',
        ),
        collect=dict(n_sample=1, ),
        other=dict(replay_buffer=dict(replay_buffer_size=1000000, ), ),
    ),
    wandb_logger=dict(
        gradient_logger=True, video_logger=True, plot_logger=True, action_logger=True, return_logger=False
    ),
)

cfg = EasyDict(cfg)

import ding.envs.gym_env
env = ding.envs.gym_env.env
