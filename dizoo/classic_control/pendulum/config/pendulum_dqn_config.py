from easydict import EasyDict

pendulum_dqn_config = dict(
    exp_name='pendulum_dqn_seed0',
    env=dict(
        collector_env_num=10,
        evaluator_env_num=5,
        # (bool) Scale output action into legal range.
        act_scale=True,
        n_evaluator_episode=5,
        stop_value=-250,
        continuous=False,
        # The path to save the game replay
        # replay_path='./pendulum_dqn_seed0/video',
    ),
    policy=dict(
        cuda=False,
        load_path='pendulum_dqn_seed0/ckpt/ckpt_best.pth.tar',  # necessary for eval
        model=dict(
            obs_shape=3,
            action_shape=11,  # mean the action shape is 11, 11 discrete actions
            encoder_hidden_size_list=[128, 128, 64],
            dueling=True,
        ),
        nstep=1,
        discount_factor=0.97,
        learn=dict(
            batch_size=64,
            learning_rate=0.001,
        ),
        collect=dict(n_sample=8),
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
pendulum_dqn_config = EasyDict(pendulum_dqn_config)
main_config = pendulum_dqn_config
pendulum_dqn_create_config = dict(
    env=dict(
        type='pendulum',
        import_names=['dizoo.classic_control.pendulum.envs.pendulum_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='dqn'),
    replay_buffer=dict(type='deque', import_names=['ding.data.buffer.deque_buffer_wrapper']),
)
pendulum_dqn_create_config = EasyDict(pendulum_dqn_create_config)
create_config = pendulum_dqn_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c pendulum_dqn_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline((main_config, create_config), seed=0)
