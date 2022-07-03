"""
ding/policy/mcts ding/data/buffer/game_buffer
cpp mcts python mcts
"""
from easydict import EasyDict
from dizoo.board_games.tictactoe.config.tictactoe_config import game_config

# debug
# collector_env_num=1
# evaluator_env_num=1

# TODO: cpp mcts now only support env_num=1, because in MCTS root nodes,
#  we must assign the one same action mask,
#  but when env_num>1, the action mask for different env may be different.
collector_env_num = 32
evaluator_env_num = 5
tictactoe_efficientzero_config = dict(
    exp_name='data_ez_ptree/tictactoe_efficientzero_seed0',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        stop_value=2,
        # 'one_player_mode' when eval, 'two_player_mode' when collect
        # automatically assign in tictactoe env
        # battle_mode='two_player_mode',
        battle_mode='one_player_mode',
    ),
    policy=dict(
        env_name='tictactoe',
        # TODO(pu): how to pass into game_config, which is class, not a dict
        # game_config=game_config,
        # Whether to use cuda for network.
        cuda=True,
        model=dict(
            model_type='tictactoe',
            # observation_shape=(3, 3, 3),
            observation_shape=(12, 3, 3),  # if stacked_observations=4
            action_space_size=9,
            downsample=False,
            num_blocks=1,
            num_channels=12,
            reduced_channels_reward=16,
            reduced_channels_value=16,
            reduced_channels_policy=16,
            fc_reward_layers=[8],
            fc_value_layers=[8],
            fc_policy_layers=[8],
            reward_support_size=21,
            value_support_size=21,
            lstm_hidden_size=64,
            bn_mt=0.1,
            proj_hid=128,
            proj_out=128,
            pred_hid=64,
            pred_out=128,
            init_zero=True,
            state_norm=False,
        ),
        # learn_mode config
        learn=dict(
            update_per_collect=10,
            batch_size=64,
            learning_rate=0.002,
            # Frequency of target network update.
            target_update_freq=200,
        ),
        # collect_mode config
        collect=dict(
            # You can use either "n_sample" or "n_episode" in collector.collect.
            # Get "n_sample" samples per collect.
            n_episode=collector_env_num,
        ),
        # command_mode config
        other=dict(
            # Epsilon greedy with decay.
            eps=dict(
                # Decay type. Support ['exp', 'linear'].
                type='exp',
                start=0.95,
                end=0.1,
                decay=int(5e4),
            ),
            replay_buffer=dict(replay_buffer_size=int(3e3), type='game')
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
    policy=dict(type='efficientzero'),
    collector=dict(type='episode_muzero', get_train_sample=True)
)
tictactoe_efficientzero_create_config = EasyDict(tictactoe_efficientzero_create_config)
create_config = tictactoe_efficientzero_create_config

if __name__ == "__main__":
    from ding.entry import serial_pipeline_muzero
    serial_pipeline_muzero([main_config, create_config], game_config=game_config, seed=0, max_env_step=int(5e6))