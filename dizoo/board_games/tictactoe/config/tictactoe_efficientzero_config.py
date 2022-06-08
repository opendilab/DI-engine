from easydict import EasyDict
from tictactoe_config import game_config

nstep = 3
tictactoe_efficientzero_config = dict(
    exp_name='tictactoe_efficientzero_seed0',
    env=dict(
        # Whether to use shared memory. Only effective if "env_manager_type" is 'subprocess'
        # Env number respectively for collector and evaluator.
        collector_env_num=1,
        evaluator_env_num=1,
        n_evaluator_episode=1,
        stop_value=200,
    ),
    policy=dict(
        # Whether to use cuda for network.
        cuda=False,
        model=dict(
            action_space_size=9,
            num_blocks=1,
            observation_shape=(3, 3, 3),
            num_channels=12,
            reduced_channels_reward=16,
            reduced_channels_value=16,
            reduced_channels_policy=16,
            fc_reward_layers=[32],
            fc_value_layers=[32],
            fc_policy_layers=[32],
            reward_support_size=601,
            value_support_size=601,
            downsample=False,
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
            batch_size=2,  # TODO(pu)
            learning_rate=0.001,
            # Frequency of target network update.
            target_update_freq=100,
        ),
        # collect_mode config
        collect=dict(
            # You can use either "n_sample" or "n_episode" in collector.collect.
            # Get "n_sample" samples per collect.
            n_episode=8,
            # Cut trajectories into pieces with length "unroll_len".
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
tictactoe_efficientzero_config = EasyDict(tictactoe_efficientzero_config)
main_config = tictactoe_efficientzero_config

tictactoe_efficientzero_create_config = dict(
    env=dict(
        type='tictactoe',
        import_names=['dizoo.board_games.tictactoe.envs.tictactoe_env'],
    ),
    # env_manager=dict(type='subprocess'),
    env_manager=dict(type='base'),
    policy=dict(type='muzero'),
    collector=dict(type='episode_muzero', get_train_sample=True)
)
tictactoe_efficientzero_create_config = EasyDict(tictactoe_efficientzero_create_config)
create_config = tictactoe_efficientzero_create_config

if __name__ == "__main__":
    from ding.entry import serial_pipeline_muzero
    serial_pipeline_muzero([main_config, create_config], game_config=game_config, seed=0)
