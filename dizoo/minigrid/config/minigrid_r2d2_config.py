from easydict import EasyDict
from ding.entry import serial_pipeline
collector_env_num = 8
evaluator_env_num = 5
minigrid_r2d2_config = dict(
    exp_name='minigrid_empty8_r2d2_bs2_n2_ul40_upc4_tuf200_ed1e4_rbs5e4',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        env_id='MiniGrid-Empty-8x8-v0',
        n_evaluator_episode=5,
        stop_value=0.96,
    ),
    policy=dict(
        cuda=False,
        on_policy=False,
        priority=False,
        model=dict(
            obs_shape=2739,
            action_shape=7,
            encoder_hidden_size_list=[256, 128, 64, 64],
        ),
        discount_factor=0.997,
        burnin_step=2,
        nstep=2,
        unroll_len=40,
        learn=dict(
            update_per_collect=4, #20
            batch_size=64,
            learning_rate=0.0005,
            target_update_freq=200, #2500,
        ),
        collect=dict(
            n_sample=32,
            env_num=collector_env_num,
        ),
        eval=dict(env_num=evaluator_env_num, ),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.05,
                decay=10000,
            ), replay_buffer=dict(replay_buffer_size=50000, )
        ),
    ),
)
minigrid_r2d2_config = EasyDict(minigrid_r2d2_config)
main_config = minigrid_r2d2_config
minigrid_r2d2_create_config = dict(
    env=dict(
        type='minigrid',
        import_names=['dizoo.minigrid.envs.minigrid_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='r2d2'),
)
minigrid_r2d2_create_config = EasyDict(minigrid_r2d2_create_config)
create_config = minigrid_r2d2_create_config

if __name__ == "__main__":
    serial_pipeline([main_config, create_config], seed=0)