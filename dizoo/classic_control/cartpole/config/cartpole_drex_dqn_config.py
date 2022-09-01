from easydict import EasyDict

cartpole_drex_dqn_config = dict(
    exp_name='cartpole_drex_dqn_seed0',
    env=dict(
        manager=dict(shared_memory=True, reset_inplace=True),
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=195,
    ),
    reward_model=dict(
        type='drex',
        min_snippet_length=5,
        max_snippet_length=100,
        checkpoint_min=0,
        checkpoint_max=1000,
        checkpoint_step=1000,
        learning_rate=1e-5,
        update_per_collect=1,
        # path to expert models that generate demonstration data
        # Users should add their own model path here. Model path should lead to an exp_name.
        # Absolute path is recommended.
        # In DI-engine, it is ``exp_name``.
        # For example, if you want to use dqn to generate demos, you can use ``spaceinvaders_dqn``
        expert_model_path='expert_model_path_placeholder',
        # path to save reward model
        # Users should add their own model path here.
        # Absolute path is recommended.
        # For example, if you use ``spaceinvaders_drex``, then the reward model will be saved in this directory.
        reward_model_path='reward_model_path_placeholder + ./spaceinvaders.params',
        # path to save generated observations.
        # Users should add their own model path here.
        # Absolute path is recommended.
        # For example, if you use ``spaceinvaders_drex``, then all the generated data will be saved in this directory.
        offline_data_path='offline_data_path_placeholder',
        # path to pretrained bc model. If omitted, bc will be trained instead.
        # Users should add their own model path here. Model path should lead to a model ckpt.
        # Absolute path is recommended.
        bc_path='bc_path_placeholder',
        # list of noises
        eps_list=[0, 0.5, 1],
        num_trajs_per_bin=20,
        bc_iterations=6000,
    ),
    policy=dict(
        cuda=False,
        model=dict(
            obs_shape=4,
            action_shape=2,
            encoder_hidden_size_list=[128, 128, 64],
            dueling=True,
        ),
        nstep=1,
        discount_factor=0.97,
        learn=dict(
            batch_size=64,
            learning_rate=0.001,
        ),
        collect=dict(n_sample=8, collector=dict(get_train_sample=False, )),
        eval=dict(evaluator=dict(eval_freq=40, )),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.1,
                decay=10000,
            ),
            replay_buffer=dict(replay_buffer_size=20000, ),
        ),
    ),
)
cartpole_drex_dqn_config = EasyDict(cartpole_drex_dqn_config)
main_config = cartpole_drex_dqn_config
cartpole_drex_dqn_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='dqn'),
)
cartpole_drex_dqn_create_config = EasyDict(cartpole_drex_dqn_create_config)
create_config = cartpole_drex_dqn_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c cartpole_drex_dqn_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline((main_config, create_config), seed=0)
