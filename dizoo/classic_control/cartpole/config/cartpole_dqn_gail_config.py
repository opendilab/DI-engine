from easydict import EasyDict

cartpole_dqn_gail_config = dict(
    exp_name='cartpole_dqn_gail_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=195,
    ),
    reward_model=dict(
        type='gail',
        input_size=5,
        hidden_size=64,
        batch_size=64,
        learning_rate=1e-3,
        update_per_collect=100,
        # Users should add their own model path here. Model path should lead to a model.
        # Absolute path is recommended.
        # In DI-engine, it is ``exp_name/ckpt/ckpt_best.pth.tar``.
        # If collect_data is True, we will use this expert_model_path to collect expert data first, rather than we
        # will load data directly from user-defined data_path
        expert_model_path='model_path_placeholder',
        collect_count=1000,
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
            update_per_collect=3,
        ),
        collect=dict(n_sample=64),
        eval=dict(evaluator=dict(eval_freq=10, )),
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
cartpole_dqn_gail_config = EasyDict(cartpole_dqn_gail_config)
main_config = cartpole_dqn_gail_config
cartpole_dqn_gail_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='dqn'),
)
cartpole_dqn_gail_create_config = EasyDict(cartpole_dqn_gail_create_config)
create_config = cartpole_dqn_gail_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial_gail -c cartpole_dqn_gail_config.py -s 0`
    # then input the config you used to generate your expert model in the path mentioned above
    # e.g. cartpole_dqn_config.py
    from ding.entry import serial_pipeline_gail
    from dizoo.classic_control.cartpole.config import cartpole_dqn_config, cartpole_dqn_create_config
    expert_main_config = cartpole_dqn_config
    expert_create_config = cartpole_dqn_create_config
    serial_pipeline_gail(
        (main_config, create_config), (expert_main_config, expert_create_config),
        max_env_step=1000000,
        seed=0,
        collect_data=True
    )
