from easydict import EasyDict
from ding.entry import serial_pipeline

collector_env_num = 8
evaluator_env_num = 1
memory_len_r2d2_gtrxl_config = dict(
    exp_name='memory_len_22_r2d2_gtrxl',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=100,
        env_id='memory_len/22',  # 100 memory steps, 101 obs per episode
        stop_value=1.,
    ),
    policy=dict(
        cuda=True,
        priority=True,
        priority_IS_weight=True,
        model=dict(
            obs_shape=3,
            action_shape=2,
            memory_len=2,
            embedding_dim=64,
            gru_bias=1.
        ),
        discount_factor=0.997,
        nstep=2,
        # (int) the whole sequence length to unroll the RNN network minus
        # the timesteps of burnin part,
        # i.e., <the whole sequence length> = <burnin_step> + <unroll_len>
        unroll_len=103,
        seq_len=103,
        learn=dict(
            update_per_collect=8,
            batch_size=64,
            learning_rate=0.0005,
            target_update_theta=0.001,
        ),
        collect=dict(
            # NOTE it is important that don't include key n_sample here, to make sure self._traj_len=INF
            each_iter_n_sample=32,
            env_num=collector_env_num,
        ),
        eval=dict(env_num=evaluator_env_num, evaluator=dict(eval_freq=100, )),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.05,
                decay=1e5,
            ),
            replay_buffer=dict(replay_buffer_size=50000,
                               # (Float type) How much prioritization is used: 0 means no prioritization while 1 means full prioritization
                               alpha=0.6,
                               # (Float type)  How much correction is used: 0 means no correction while 1 means full correction
                               beta=0.4,
                               )
        ),
    ),
)
memory_len_r2d2_gtrxl_config = EasyDict(memory_len_r2d2_gtrxl_config)
main_config = memory_len_r2d2_gtrxl_config
memory_len_r2d2_gtrxl_create_config = dict(
    env=dict(
        type='bsuite',
        import_names=['dizoo.bsuite.envs.bsuite_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='r2d2_gtrxl'),
)
memory_len_r2d2_gtrxl_create_config = EasyDict(memory_len_r2d2_gtrxl_create_config)
create_config = memory_len_r2d2_gtrxl_create_config

if __name__ == "__main__":
    serial_pipeline([main_config, create_config], seed=0)
