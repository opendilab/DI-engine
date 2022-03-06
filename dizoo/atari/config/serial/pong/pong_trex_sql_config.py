from copy import deepcopy
from easydict import EasyDict

pong_trex_sql_config = dict(
    exp_name='pong_trex_sql',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=20,
        env_id='PongNoFrameskip-v4',
        frame_stack=4,
        manager=dict(shared_memory=False, reset_inplace=True)
    ),
    reward_model=dict(
        type='trex',
        algo_for_model='sql',
        env_id='PongNoFrameskip-v4',
        min_snippet_length=50,
        max_snippet_length=100,
        checkpoint_min=10000,
        checkpoint_max=50000,
        checkpoint_step=10000,
        learning_rate=1e-5,
        update_per_collect=1,
        expert_model_path='abs model path',
        reward_model_path='abs data path + ./pong.params',
        offline_data_path='abs data path',
    ),
    policy=dict(
        cuda=False,
        priority=False,
        model=dict(
            obs_shape=[4, 84, 84],
            action_shape=6,
            encoder_hidden_size_list=[128, 128, 512],
        ),
        nstep=1,
        discount_factor=0.99,
        learn=dict(update_per_collect=10, batch_size=32, learning_rate=0.0001, target_update_freq=500, alpha=0.12),
        collect=dict(n_sample=96, ),
        other=dict(
            eps=dict(
                type='exp',
                start=1.,
                end=0.05,
                decay=250000,
            ),
            replay_buffer=dict(replay_buffer_size=100000, ),
        ),
    ),
)
pong_trex_sql_config = EasyDict(pong_trex_sql_config)
main_config = pong_trex_sql_config
pong_trex_sql_create_config = dict(
    env=dict(
        type='atari',
        import_names=['dizoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='sql'),
)
pong_trex_sql_create_config = EasyDict(pong_trex_sql_create_config)
create_config = pong_trex_sql_create_config
