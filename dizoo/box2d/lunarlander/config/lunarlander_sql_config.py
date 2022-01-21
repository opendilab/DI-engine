from easydict import EasyDict

lunarlander_sql_config = dict(
    exp_name='lunarlander_sql',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        env_id='LunarLander-v2',
        n_evaluator_episode=5,
        stop_value=200,
    ),
    policy=dict(
        cuda=False,
        model=dict(
            obs_shape=8,
            action_shape=4,
            encoder_hidden_size_list=[128, 128, 64],
            dueling=True,
        ),
        nstep=1,
        discount_factor=0.97,
        learn=dict(batch_size=64, learning_rate=0.001, alpha=0.08),
        collect=dict(n_sample=64),
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
lunarlander_sql_config = EasyDict(lunarlander_sql_config)
main_config = lunarlander_sql_config
lunarlander_sql_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['dizoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='sql'),
)
lunarlander_sql_create_config = EasyDict(lunarlander_sql_create_config)
create_config = lunarlander_sql_create_config
