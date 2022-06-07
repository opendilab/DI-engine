from easydict import EasyDict
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
        stop_value=200,

        env_name='PongNoFrameskip-v4',
        obs_shape=(12, 96, 96),
        gray_scale=False,

        training_steps=100000,
        last_steps=20000,
        test_interval=10000,
        log_interval=1000,
        vis_interval=1000,
        # test_episodes=32,
        test_episodes=2,  # TODO Debug
        checkpoint_interval=100,
        target_model_interval=200,
        save_ckpt_interval=10000,
        # max_moves=108000,
        # test_max_moves=12000,
        max_moves=100,  # TODO Debug
        test_max_moves=100,
        # history_length=400,
        history_length=10,  # TODO Debug
        discount=0.997,
        dirichlet_alpha=0.3,
        value_delta_max=0.01,
        # num_simulations=50,
        num_simulations=2,  # TODO Debug
        # batch_size=256,
        batch_size=2,  # TODO Debug
        td_steps=5,
        num_actors=1,
        # network initialization/ & normalization
        episode_life=True,
        init_zero=True,
        clip_reward=True,
        # storage efficient
        # cvt_string=True,
        cvt_string=False,  # TODO
        image_based=True,
        # lr scheduler
        lr_warm_up=0.01,
        lr_init=0.2,
        lr_decay_rate=0.1,
        lr_decay_steps=100000,
        auto_td_steps_ratio=0.3,
        # replay window
        start_transitions=8,
        total_transitions=int(1e6),  # TODO Debug
        # frame skip & stack observation
        frame_skip=4,
        stacked_observations=4,
        # coefficient
        reward_loss_coeff=1,
        value_loss_coeff=0.25,
        policy_loss_coeff=1,
        consistency_coeff=2,
        # reward sum
        lstm_hidden_size=512,
        lstm_horizon_len=5,
        # siamese
        proj_hid=1024,
        proj_out=1024,
        pred_hid=512,
        pred_out=1024,
    ),
    policy=dict(
        # Whether to use cuda for network.
        cuda=False,
        device='cpu',
        on_policy=False,
        priority=True,
        priority_IS_weight=True,
        batch_size=256,
        discount_factor=0.997,
        learning_rate=1e-3,
        momentum=0.9,
        weight_decay=1e-4,
        grad_clip_type='clip_norm',
        grad_clip_value=5,
        policy_weight=1.0,
        value_weight=0.25,
        consistent_weight=1.0,
        value_prefix_weight=2.0,
        image_unroll_len=5,
        lstm_horizon_len=5,
        # collect
        # collect_env_num=8,
        # action_shape=6,
        simulation_num=50,
        root_dirichlet_alpha=0.3,
        root_exploration_fraction=0.25,
        value_delta_max=0.01,

        # UCB formula
        pb_c_base = 19652,
        pb_c_init = 1.25,
        discount=0.997,
        # num_simulations=50,
        num_simulations=2,  # TODO Debug
        amp_type = 'torch_amp',

        model=dict(
            observation_shape=(12,96,96),
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
            # inverse_value_transform=game_config.inverse_value_transform,
            # inverse_reward_transform=game_config.inverse_reward_transform,
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
            multi_gpu=False,
            update_per_collect=10,
            batch_size=64,
            learning_rate=0.001,
            # Frequency of target network update.
            target_update_freq=100,
            # grad_clip_type='clip_norm',
            # grad_clip_value=0.5,

        ),
        # collect_mode config
        collect=dict(
            # You can use either "n_sample" or "n_episode" in collector.collect.
            # Get "n_sample" samples per collect.
            n_sample=64,
            # Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
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
        type='atari-di',
        import_names=['dizoo.board_games.atari.envs.atari_env_di'],
    ),
    # env_manager=dict(type='subprocess'),
    env_manager=dict(type='base'),
    policy=dict(type='muzero'),
    # collector=dict(type='sample_muzero', )
)
atari_efficientzero_create_config = EasyDict(atari_efficientzero_create_config)
create_config = atari_efficientzero_create_config

if __name__ == "__main__":
    from ding.entry import serial_pipeline_muzero
    serial_pipeline_muzero([main_config, create_config], game_config=game_config, seed=0)
