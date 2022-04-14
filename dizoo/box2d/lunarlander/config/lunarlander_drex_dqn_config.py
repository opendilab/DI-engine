from easydict import EasyDict

nstep = 1
lunarlander_trex_dqn_default_config = dict(
    exp_name='lunarlander_drex_seed0',
    env=dict(
        # Whether to use shared memory. Only effective if "env_manager_type" is 'subprocess'
        manager=dict(shared_memory=True, force_reproducibility=True),
        # Env number respectively for collector and evaluator.
        collector_env_num=8,
        evaluator_env_num=5,
        env_id='LunarLander-v2',
        n_evaluator_episode=5,
        stop_value=200,
    ),
    reward_model=dict(
        type='drex',
        algo_for_model='dqn',
        env_id='LunarLander-v2',
        min_snippet_length=30,
        max_snippet_length=100,
        num_snippets=60000,
        learning_rate=1e-5,
        update_per_collect=1,
        # path to expert models that generate demonstration data
        # Users should add their own model path here. Model path should lead to an exp_name.
        # Absolute path is recommended.
        # In DI-engine, it is ``exp_name``.
        # For example, if you want to use dqn to generate demos, you can use ``lunarlander_dqn``
        expert_model_path='expert_model_path_placeholder',
        # path to save reward model
        # Users should add their own model path here.
        # Absolute path is recommended.
        # For example, if you use ``lunarlander_drex``, then the reward model will be saved in this directory.
        reward_model_path='reward_model_path_placeholder + ./spaceinvaders.params',
        # path to save generated observations.
        # Users should add their own model path here.
        # Absolute path is recommended.
        # For example, if you use ``lunarlander_drex``, then all the generated data will be saved in this directory.
        offline_data_path='offline_data_path_placeholder',
        # path to pretrained bc model. If omitted, bc will be trained instead.
        # Users should add their own model path here. Model path should lead to a model ckpt.
        # Absolute path is recommended.
        # list of noises
        eps_list=[0, 0.5, 1],
        num_trajs_per_bin=20,
        bc_iterations=50000,
    ),
    policy=dict(
        # Whether to use cuda for network.
        cuda=False,
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
            collector=dict(get_train_sample=False, )
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
lunarlander_drex_dqn_default_config = EasyDict(lunarlander_trex_dqn_default_config)
main_config = lunarlander_drex_dqn_default_config

lunarlander_drex_dqn_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['dizoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='dqn'),
)
lunarlander_drex_dqn_create_config = EasyDict(lunarlander_drex_dqn_create_config)
create_config = lunarlander_drex_dqn_create_config

if __name__ == '__main__':
    from ding.entry import drex_collecting_data, serial_pipeline_reward_model_preference_based_irl

    args = EasyDict(dict(cfg=[main_config, create_config], seed=0, device='cuda'))
    drex_collecting_data(args)
    serial_pipeline_reward_model_preference_based_irl([main_config, create_config], seed=0)
