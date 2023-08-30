from easydict import EasyDict
import ding.envs.gym_env

cfg = dict(
    exp_name='LunarLander-v2-PG',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=8,
        env_id='LunarLander-v2',
        n_evaluator_episode=8,
        stop_value=260,
    ),
    policy=dict(
        cuda=True,
        model=dict(
            obs_shape=8,
            action_shape=4,
        ),
        learn=dict(
            batch_size=320,
            learning_rate=3e-4,
            entropy_weight=0.001,
            grad_norm=0.5,
        ),
        collect=dict(
            n_episode=8,
            discount_factor=0.99,
        ),
        eval=dict(evaluator=dict(eval_freq=1000, ), ),
    ),
    wandb_logger=dict(
        gradient_logger=True, video_logger=True, plot_logger=True, action_logger=True, return_logger=False
    ),
)

cfg = EasyDict(cfg)

env = ding.envs.gym_env.env
