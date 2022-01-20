from ding.entry import serial_pipeline
from easydict import EasyDict

qbert_r2d2_gtrxl_config = dict(
    exp_name='qbert_r2d2_gtrxl',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=10000000000,
        env_id='QbertNoFrameskip-v4',
        frame_stack=4,
        manager=dict(shared_memory=False, )
    ),
    policy=dict(
        cuda=True,
        priority=True,
        priority_IS_weight=True,
        model=dict(
            obs_shape=[4, 84, 84],
            action_shape=6,
            encoder_hidden_size_list=[128, 256, 1024],
            hidden_size=1024,
            gru_bias=1.,
            memory_len=0,
        ),
        discount_factor=0.99,
        burnin_step=0,
        nstep=3,
        unroll_len=13,
        seq_len=10,
        learn=dict(
            update_per_collect=8,
            batch_size=64,
            learning_rate=0.0005,
            target_update_theta=0.001,
        ),
        collect=dict(
            # NOTE it is important that don't include key n_sample here, to make sure self._traj_len=INF
            each_iter_n_sample=32,
            env_num=8,
        ),
        eval=dict(env_num=8, ),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.05,
                decay=1e5,
            ),
            replay_buffer=dict(
                replay_buffer_size=10000,
                # (Float type) How much prioritization is used: 0 means no prioritization while 1 means full prioritization
                alpha=0.6,
                # (Float type)  How much correction is used: 0 means no correction while 1 means full correction
                beta=0.4,
            )
        ),
    ),
)
qbert_r2d2_gtrxl_config = EasyDict(qbert_r2d2_gtrxl_config)
main_config = qbert_r2d2_gtrxl_config
qbert_r2d2_gtrxl_create_config = dict(
    env=dict(
        type='atari',
        import_names=['dizoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='r2d2_gtrxl'),
)
qbert_r2d2_gtrxl_create_config = EasyDict(qbert_r2d2_gtrxl_create_config)
create_config = qbert_r2d2_gtrxl_create_config

if __name__ == '__main__':
    serial_pipeline((main_config, create_config), seed=0)
