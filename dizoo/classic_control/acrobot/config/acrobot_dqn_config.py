from easydict import EasyDict

acrobot_dqn_config = dict(
    exp_name='acrobot_dqn_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=-60,
        env_id='Acrobot-v1',
        replay_path='acrobot_dqn_seed0/video',
    ),
    policy=dict(
        cuda=True,
        model=dict(
            obs_shape=6,
            action_shape=3,
            encoder_hidden_size_list=[256, 256],
            dueling=True,
        ),
        nstep=3,
        discount_factor=0.99,
        learn=dict(
            update_per_collect=10,
            batch_size=128,
            learning_rate=0.0001,
            target_update_freq=250,
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
acrobot_dqn_config = EasyDict(acrobot_dqn_config)
main_config = acrobot_dqn_config
acrobot_dqn_create_config = dict(
    env=dict(type='acrobot', import_names=['dizoo.classic_control.acrobot.envs.acrobot_env']),
    env_manager=dict(type='subprocess'),
    policy=dict(type='dqn'),
    replay_buffer=dict(type='deque', import_names=['ding.data.buffer.deque_buffer_wrapper']),
)
acrobot_dqn_create_config = EasyDict(acrobot_dqn_create_config)
create_config = acrobot_dqn_create_config

if __name__ == "__main__":
    from ding.entry import serial_pipeline
    serial_pipeline((main_config, create_config), seed=0)
