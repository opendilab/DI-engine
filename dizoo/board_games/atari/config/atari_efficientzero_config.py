from easydict import EasyDict
# from dizoo.board_games.atari.config.atari_config_dict import game_config
from dizoo.board_games.atari.config.atari_config import game_config


nstep = 3
atari_efficientzero_config = dict(
    exp_name='atari_efficientzero_seed0',
    env=dict(
        # Whether to use shared memory. Only effective if "env_manager_type" is 'subprocess'
        # Env number respectively for collector and evaluator.
        collector_env_num=2,
        evaluator_env_num=2,
        n_evaluator_episode=2,
        stop_value=20,
        env_name='PongNoFrameskip-v4',
        obs_shape=(12, 96, 96),
        gray_scale=False,
        max_moves=100,  # TODO
        discount=0.997,
        episode_life=True,
        cvt_string=False,  # TODO
        frame_skip=4,
    ),
    policy=dict(
        env_name='PongNoFrameskip-v4',
        # TODO(pu): how to pass into game_config, which is class, not a dict
        # game_config=game_config,
        # Whether to use cuda for network.
        cuda=False,
        model=dict(
            observation_shape=(12, 96, 96),  # 3,96,96 stack4
            action_space_size=6,
            num_blocks=1,
            num_channels=64,
            reduced_channels_reward=16,
            reduced_channels_value=16,
            reduced_channels_policy=16,
            fc_reward_layers=[32],
            fc_value_layers=[32],
            fc_policy_layers=[32],
            reward_support_size=601,
            value_support_size=601,
            downsample=True,
            lstm_hidden_size=512,
            bn_mt=0.1,
            proj_hid=1024,
            proj_out=1024,
            pred_hid=512,
            pred_out=1024,
            init_zero=True,
            state_norm=False,
        ),
        # learn_mode config
        learn=dict(
            update_per_collect=10,
            batch_size=64,  # TODO(pu)
            learning_rate=0.001,
            # Frequency of target network update.
            target_update_freq=100,

        ),
        # collect_mode config
        collect=dict(
            # You can use either "n_sample" or "n_episode" in collector.collect.
            # Get "n_sample" samples per collect.
            n_episode=8,
        ),
        # command_mode config
        other=dict(
            # Epsilon greedy with decay.
            eps=dict(
                # Decay type. Support ['exp', 'linear'].
                type='exp',
                start=0.95,
                end=0.1,
                decay=50000,
            ),
            replay_buffer=dict(replay_buffer_size=100000, type='game')
        ),
    ),
)
atari_efficientzero_config = EasyDict(atari_efficientzero_config)
main_config = atari_efficientzero_config

atari_efficientzero_create_config = dict(
    env=dict(
        type='atari-game',
        import_names=['dizoo.board_games.atari.envs.atari_env_game'],
    ),
    # env_manager=dict(type='subprocess'),
    env_manager=dict(type='base'),
    policy=dict(type='efficientzero'),
    collector=dict(type='episode_muzero', get_train_sample=True)
)
atari_efficientzero_create_config = EasyDict(atari_efficientzero_create_config)
create_config = atari_efficientzero_create_config

if __name__ == "__main__":
    from ding.entry import serial_pipeline_muzero
    serial_pipeline_muzero([main_config, create_config],  seed=0, max_env_step=int(10e6), game_config=game_config)
