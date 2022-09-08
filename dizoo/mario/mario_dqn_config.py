from easydict import EasyDict

mario_dqn_config = dict(
    exp_name='mario_dqn_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=100000,
        replay_path='mario_dqn_seed0/video',
    ),
    policy=dict(
        cuda=True,
        model=dict(
            obs_shape=[4, 84, 84],
            action_shape=2,
            encoder_hidden_size_list=[128, 128, 256],
            dueling=True,
        ),
        nstep=3,
        discount_factor=0.99,
        learn=dict(
            update_per_collect=10,
            batch_size=32,
            learning_rate=0.0001,
            target_update_freq=500,
        ),
        collect=dict(n_sample=96, ),
        eval=dict(evaluator=dict(eval_freq=2000, )),
        other=dict(
            eps=dict(
                type='exp',
                start=1.,
                end=0.05,
                decay=250000,
            ),
            replay_buffer=dict(replay_buffer_size=100000, ),
        ),
    ),
)
mario_dqn_config = EasyDict(mario_dqn_config)
main_config = mario_dqn_config
mario_dqn_create_config = dict(
    env_manager=dict(type='subprocess'),
    policy=dict(type='dqn'),
)
mario_dqn_create_config = EasyDict(mario_dqn_create_config)
create_config = mario_dqn_create_config
# you can run `python3 -u mario_dqn_main.py`
