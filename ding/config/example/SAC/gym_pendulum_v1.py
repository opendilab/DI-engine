from easydict import EasyDict
import ding.envs.gym_env

cfg = dict(
    exp_name='Pendulum-v1-SAC',
    seed=0,
    env=dict(
        env_id='Pendulum-v1',
        collector_env_num=10,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=-250,
        act_scale=True,
    ),
    policy=dict(
        cuda=True,
        priority=False,
        random_collect_size=1000,
        model=dict(
            obs_shape=3,
            action_shape=1,
            twin_critic=True,
            action_space='reparameterization',
            actor_head_hidden_size=128,
            critic_head_hidden_size=128,
        ),
        learn=dict(
            update_per_collect=1,
            batch_size=128,
            learning_rate_q=0.001,
            learning_rate_policy=0.001,
            learning_rate_alpha=0.0003,
            ignore_done=True,
            target_theta=0.005,
            discount_factor=0.99,
            auto_alpha=True,
        ),
        collect=dict(n_sample=10, ),
        eval=dict(evaluator=dict(eval_freq=100, )),
        other=dict(replay_buffer=dict(replay_buffer_size=100000, ), ),
    ),
    wandb_logger=dict(
        gradient_logger=True, video_logger=True, plot_logger=True, action_logger=True, return_logger=False
    ),
)

cfg = EasyDict(cfg)

env = ding.envs.gym_env.env
