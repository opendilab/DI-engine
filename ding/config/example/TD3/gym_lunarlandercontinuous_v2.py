from easydict import EasyDict
from functools import partial
import ding.envs.gym_env

cfg = dict(
    exp_name='LunarLanderContinuous-V2-TD3',
    seed=0,
    env=dict(
        env_id='LunarLanderContinuous-v2',
        collector_env_num=4,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=240,
        act_scale=True,
    ),
    policy=dict(
        cuda=True,
        random_collect_size=10000,
        model=dict(
            obs_shape=8,
            action_shape=2,
            action_space='regression',
        ),
        learn=dict(
            update_per_collect=256,
            batch_size=256,
            learning_rate_actor=3e-4,
            learning_rate_critic=1e-3,
            noise=True,
            noise_sigma=0.1,
            noise_range=dict(
                min=-0.5,
                max=0.5,
            ),
        ),
        collect=dict(
            n_sample=256,
            noise_sigma=0.1,
        ),
        eval=dict(evaluator=dict(eval_freq=1000, ), ),
        other=dict(replay_buffer=dict(replay_buffer_size=100000, ), ),
    ),
    wandb_logger=dict(
        gradient_logger=True, video_logger=True, plot_logger=True, action_logger=True, return_logger=False
    ),
)

cfg = EasyDict(cfg)

env = partial(ding.envs.gym_env.env, continuous=True)
