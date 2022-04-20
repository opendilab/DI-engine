from easydict import EasyDict

cartpole_sql_config = dict(
    exp_name='cartpole_sql_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=195,
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
        learn=dict(batch_size=64, learning_rate=0.001, alpha=0.12),
        collect=dict(n_sample=8),
        eval=dict(evaluator=dict(eval_freq=50, )),  # note: this is the times after which you learns to evaluate
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
cartpole_sql_config = EasyDict(cartpole_sql_config)
main_config = cartpole_sql_config
cartpole_sql_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='sql'),
)
cartpole_sql_create_config = EasyDict(cartpole_sql_create_config)
create_config = cartpole_sql_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c cartpole_sql_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline((main_config, create_config), seed=0)
