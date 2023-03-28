from easydict import EasyDict

nstep = 3
carracing_dqn_config = dict(
    exp_name='carracing_dqn_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=8,
        env_id='CarRacing-v2',
        continuous=False,
        n_evaluator_episode=8,
        stop_value=900,
        # replay_path='./carracing_dqn_seed0/video',
    ),
    policy=dict(
        cuda=True,
        # load_path='carracing_dqn_seed0/ckpt/ckpt_best.pth.tar',
        model=dict(
            obs_shape=[3, 96, 96],
            action_shape=5,
            encoder_hidden_size_list=[64, 64, 128],
            dueling=True,
        ),
        discount_factor=0.99,
        nstep=nstep,
        learn=dict(
            update_per_collect=10,
            batch_size=64,
            learning_rate=0.0001,
            target_update_freq=100,
        ),
        collect=dict(n_sample=64, ),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.1,
                decay=50000,
            ), replay_buffer=dict(replay_buffer_size=100000, )
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
    policy=dict(type='dqn'),
)
carracing_dqn_create_config = EasyDict(carracing_dqn_create_config)
create_config = carracing_dqn_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c carracing_dqn_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline([main_config, create_config], seed=0)
