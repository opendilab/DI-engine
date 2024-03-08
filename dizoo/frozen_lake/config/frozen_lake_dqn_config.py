from easydict import EasyDict

frozen_lake_dqn_config = dict(
    exp_name='frozen_lake_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=50,
        stop_value=5,
        env_id = 'FrozenLake-v1',
        desc = None,
        map_name = "4x4",
        is_slippery = True,
        save_replay_gif = True,
        save_replay_path='frozen_lake_seed0/video',

    ),

    policy = dict(
        cuda=True,
        load_path='cartpole_dqn_seed0/ckpt/ckpt_best.pth.tar',
        model = dict(
            obs_shape=16,
            action_shape=1,
            encoder_hidden_size_list=[128, 128, 64],
            dueling=True,
        ),
        nstep = 3,
        discount_factor=0.97,
        learn=dict(
            update_per_collect=5,
            batch_size=64,
            learning_rate=0.001,
        ),
        collect = dict(n_sample=8),
        eval=dict(evaluator=dict(eval_freq=40, )),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.1,
                decay=10000,
            ),
            replay_buffer=dict(replay_buffer_size=128, ),
        ),
    ),
)

frozen_lake_dqn_config = EasyDict(frozen_lake_dqn_config)
main_config = frozen_lake_dqn_config

frozen_lake_dqn_config = dict(
    env=dict(
        type='frozen_lake',
        import_names=['dizoo.frozen_lake.envs.frozen_lake_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='dqn'),
    replay_buffer=dict(type='deque', import_names=['ding.data.buffer.deque_buffer_wrapper']),
)

frozen_lake_dqn_config = EasyDict(frozen_lake_dqn_config)
create_config = frozen_lake_dqn_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c cartpole_dqn_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline((main_config, create_config), seed=0)