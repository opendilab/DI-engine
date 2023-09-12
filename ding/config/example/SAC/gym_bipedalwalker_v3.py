from easydict import EasyDict
import ding.envs.gym_env

cfg = dict(
    exp_name='BipedalWalker-v3-SAC',
    seed=0,
    env=dict(
        env_id='BipedalWalker-v3',
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        act_scale=True,
        rew_clip=True,
    ),
    policy=dict(
        cuda=True,
        random_collect_size=10000,
        model=dict(
            obs_shape=24,
            action_shape=4,
            twin_critic=True,
            action_space='reparameterization',
            actor_head_hidden_size=128,
            critic_head_hidden_size=128,
        ),
        learn=dict(
            update_per_collect=64,
            batch_size=256,
            learning_rate_q=0.0003,
            learning_rate_policy=0.0003,
            learning_rate_alpha=0.0003,
            target_theta=0.005,
            discount_factor=0.99,
            auto_alpha=True,
            learner=dict(hook=dict(log_show_after_iter=1000, ))
        ),
        collect=dict(n_sample=64, ),
        other=dict(replay_buffer=dict(replay_buffer_size=300000, ), ),
    ),
    wandb_logger=dict(
        gradient_logger=True, video_logger=True, plot_logger=True, action_logger=True, return_logger=False
    ),
)

cfg = EasyDict(cfg)

env = ding.envs.gym_env.env
