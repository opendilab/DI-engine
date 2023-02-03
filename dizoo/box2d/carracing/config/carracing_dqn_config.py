from easydict import EasyDict

nstep = 3
carracing_dqn_config = dict(
    exp_name='carracing_dqn_seed0',
    env=dict(
        # Whether to use shared memory. Only effective if "env_manager_type" is 'subprocess'
        # Env number respectively for collector and evaluator.
        collector_env_num=8,
        evaluator_env_num=8,
        env_id='CarRacing-v2',
        continuous=False,
        n_evaluator_episode=8,
        stop_value=950,
        # The path to save the game replay
        # replay_path='./carracing_dqn_seed0/video',
    ),
    policy=dict(
        # Whether to use cuda for network.
        cuda=True,
        load_path="./carracing_seed0/ckpt/ckpt_best.pth.tar",
        model=dict(
            obs_shape=[3, 96, 96],
            action_shape=5,
            encoder_hidden_size_list=[128, 128, 512],
            # Whether to use dueling head.
            dueling=True,
        ),
        # Reward's future discount factor, aka. gamma.
        discount_factor=0.99,
        # How many steps in td error.
        nstep=nstep,
        # learn_mode config
        learn=dict(
            update_per_collect=10,
            batch_size=64,
            learning_rate=0.001,
            # Frequency of target network update.
            target_update_freq=100,
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
            replay_buffer=dict(replay_buffer_size=100000, )
        ),
    ),
)
carracing_dqn_config = EasyDict(carracing_dqn_config)
main_config = carracing_dqn_config

carracing_dqn_create_config = dict(
    env=dict(
        type='carracing',
        import_names=['dizoo.box2d.carracing.envs.carracing_env'],
    ),
    env_manager=dict(type='subprocess'),
    # env_manager=dict(type='base'),
    policy=dict(type='dqn'),
)
carracing_dqn_create_config = EasyDict(carracing_dqn_create_config)
create_config = carracing_dqn_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c lunarlander_dqn_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline([main_config, create_config], seed=0)