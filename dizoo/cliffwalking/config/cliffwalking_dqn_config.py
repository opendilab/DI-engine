from easydict import EasyDict

cliffwalking_dqn_config = dict(
    exp_name='cliffwalking_dqn_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=-13,  # the optimal value of cliffwalking env
        max_episode_steps=300,
    ),
    policy=dict(
        cuda=True,
        load_path="./cliffwalking_dqn_seed0/ckpt/ckpt_best.pth.tar",
        model=dict(
            obs_shape=48,
            action_shape=4,
            encoder_hidden_size_list=[512, 64],
        ),
        discount_factor=0.99,
        nstep=1,
        learn=dict(
            update_per_collect=10,
            batch_size=128,
            learning_rate=0.001,
            target_update_freq=100,
        ),
        collect=dict(
            n_sample=64,
            unroll_len=1,
        ),
        other=dict(
            eps=dict(
                type='linear',
                start=0.95,
                end=0.25,
                decay=50000,
            ),
            replay_buffer=dict(replay_buffer_size=100000, )
        ),
    ),
)
cliffwalking_dqn_config = EasyDict(cliffwalking_dqn_config)
main_config = cliffwalking_dqn_config

cliffwalking_dqn_create_config = dict(
    env=dict(
        type='cliffwalking',
        import_names=['dizoo.cliffwalking.envs.cliffwalking_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='dqn'),
)
cliffwalking_dqn_create_config = EasyDict(cliffwalking_dqn_create_config)
create_config = cliffwalking_dqn_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c cliffwalking_dqn_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline([main_config, create_config], seed=0)
