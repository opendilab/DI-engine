from easydict import EasyDict
import ding.envs.gym_env

cfg = dict(
    exp_name='Pendulum-v1-PG',
    seed=0,
    env=dict(
        env_id='Pendulum-v1',
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=-200,
        act_scale=True,
    ),
    policy=dict(
        cuda=False,
        action_space='continuous',
        model=dict(
            action_space='continuous',
            obs_shape=3,
            action_shape=1,
        ),
        learn=dict(
            batch_size=4000,
            learning_rate=0.001,
            entropy_weight=0.001,
        ),
        collect=dict(
            n_episode=20,
            unroll_len=1,
            discount_factor=0.99,
        ),
        eval=dict(evaluator=dict(eval_freq=1, ))
    ),
    wandb_logger=dict(
        gradient_logger=True, video_logger=True, plot_logger=True, action_logger=True, return_logger=False
    ),
)

cfg = EasyDict(cfg)

env = ding.envs.gym_env.env
