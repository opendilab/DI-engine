from easydict import EasyDict

nstep = 1
lunarlander_dqn_gail_config = dict(
    exp_name='lunarlander_dqn_gail_seed0',
    env=dict(
        # Whether to use shared memory. Only effective if "env_manager_type" is 'subprocess'
        # Env number respectively for collector and evaluator.
        collector_env_num=8,
        evaluator_env_num=8,
        env_id='LunarLander-v2',
        n_evaluator_episode=8,
        stop_value=200,
    ),
    reward_model=dict(
        type='gail',
        input_size=9,
        hidden_size=64,
        batch_size=64,
        learning_rate=1e-3,
        update_per_collect=100,
        collect_count=100000,
        # Users should add their own model path here. Model path should lead to a model.
        # Absolute path is recommended.
        # In DI-engine, it is ``exp_name/ckpt/ckpt_best.pth.tar``.
        expert_model_path='model_path_placeholder',
        # Path where to store the reward model
        reward_model_path='data_path_placeholder+/reward_model/ckpt/ckpt_best.pth.tar',
        # Users should add their own data path here. Data path should lead to a file to store data or load the stored data.
        # Absolute path is recommended.
        # In DI-engine, it is usually located in ``exp_name`` directory
        # e.g. 'exp_name/expert_data.pkl'
        data_path='data_path_placeholder',
    ),
    policy=dict(
        # Whether to use cuda for network.
        cuda=False,
        # Whether the RL algorithm is on-policy or off-policy.
        on_policy=False,
        model=dict(
            obs_shape=8,
            action_shape=4,
            encoder_hidden_size_list=[512, 64],
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
lunarlander_dqn_gail_config = EasyDict(lunarlander_dqn_gail_config)
main_config = lunarlander_dqn_gail_config

lunarlander_dqn_gail_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['dizoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='dqn'),
)
lunarlander_dqn_gail_create_config = EasyDict(lunarlander_dqn_gail_create_config)
create_config = lunarlander_dqn_gail_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial_gail -c lunarlander_dqn_gail_config.py -s 0`
    # then input the config you used to generate your expert model in the path mentioned above
    # e.g. lunarlander_dqn_config.py
    from ding.entry import serial_pipeline_gail
    from dizoo.box2d.lunarlander.config import lunarlander_dqn_config, lunarlander_dqn_create_config
    expert_main_config = lunarlander_dqn_config
    expert_create_config = lunarlander_dqn_create_config
    serial_pipeline_gail(
        [main_config, create_config], [expert_main_config, expert_create_config],
        max_env_step=1000000,
        seed=0,
        collect_data=True
    )
