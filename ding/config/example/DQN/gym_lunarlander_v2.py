from easydict import EasyDict
import ding.envs.gym_env

cfg = dict(
    exp_name='LunarLander-v2-DQN',
    seed=0,
    env=dict(
        env_id='LunarLander-v2',
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=260,
    ),
    policy=dict(
        cuda=True,
        random_collect_size=25000,
        discount_factor=0.99,
        nstep=3,
        learn=dict(
            update_per_collect=10,
            batch_size=64,
            learning_rate=0.001,
            # Frequency of target network update.
            target_update_freq=100,
        ),
        model=dict(
            obs_shape=8,
            action_shape=4,
            encoder_hidden_size_list=[512, 64],
            # Whether to use dueling head.
            dueling=True,
        ),
        collect=dict(
            n_sample=64,
            unroll_len=1,
        ),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.1,
                decay=50000,
            ), replay_buffer=dict(replay_buffer_size=100000, )
        ),
    ),
    wandb_logger=dict(
        gradient_logger=True, video_logger=True, plot_logger=True, action_logger=True, return_logger=False
    ),
)

cfg = EasyDict(cfg)

env = ding.envs.gym_env.env
