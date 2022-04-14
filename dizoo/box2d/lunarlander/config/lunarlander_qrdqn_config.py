from easydict import EasyDict

lunarlander_qrdqn_config = dict(
    exp_name='lunarlander_qrdqn_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=8,
        env_id='LunarLander-v2',
        n_evaluator_episode=8,
        stop_value=200,
    ),
    policy=dict(
        cuda=False,
        model=dict(
            obs_shape=8,
            action_shape=4,
        ),
        nstep=1,
        discount_factor=0.97,
        learn=dict(
            batch_size=64,
            learning_rate=0.001,
        ),
        collect=dict(n_sample=128, ),
        eval=dict(evaluator=dict(eval_freq=50, )),
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
lunarlander_qrdqn_config = EasyDict(lunarlander_qrdqn_config)
main_config = lunarlander_qrdqn_config
lunarlander_qrdqn_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['dizoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='qrdqn'),
)
lunarlander_qrdqn_create_config = EasyDict(lunarlander_qrdqn_create_config)
create_config = lunarlander_qrdqn_create_config

if __name__ == '__main__':
    # or you can enter `ding -m serial -c lunarlander_qrdqn_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline([main_config, create_config], seed=0)
