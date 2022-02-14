from easydict import EasyDict

collector_env_num = 8
evaluator_env_num = 5
cartpole_r2d2_gtrxl_config = dict(
    exp_name='cartpole_r2d2_gtrxl',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=5,
        stop_value=195,
    ),
    policy=dict(
        cuda=False,
        priority=False,
        priority_IS_weight=False,
        model=dict(
            obs_shape=4,
            action_shape=2,
            memory_len=5,  # length of transformer memory (can be 0)
            hidden_size=256,
            gru_bias=1.,
            att_layer_num=3,
            dropout=0.,
            att_head_num=2,
        ),
        discount_factor=0.99,
        nstep=3,
        burnin_step=4,  # how many steps use to initialize the memory (can be 0)
        unroll_len=11,  # trajectory len
        seq_len=8,  # transformer input segment
        # training sequence: unroll_len - burnin_step - nstep
        learn=dict(
            update_per_collect=8,
            batch_size=64,
            learning_rate=0.0005,
            target_update_freq=500,
            value_rescale=True,
            init_memory='old',  # 'zero' or 'old', how to initialize the memory
        ),
        collect=dict(
            n_sample=32,
            env_num=collector_env_num,
        ),
        eval=dict(env_num=evaluator_env_num, evaluator=dict(eval_freq=20)),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.05,
                decay=10000,
            ), replay_buffer=dict(replay_buffer_size=100000, )
        ),
    ),
)
cartpole_r2d2_gtrxl_config = EasyDict(cartpole_r2d2_gtrxl_config)
main_config = cartpole_r2d2_gtrxl_config
cartpole_r2d2_gtrxl_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='r2d2_gtrxl'),
)
cartpole_r2d2_gtrxl_create_config = EasyDict(cartpole_r2d2_gtrxl_create_config)
create_config = cartpole_r2d2_gtrxl_create_config

if __name__ == "__main__":
    from ding.entry import serial_pipeline
    serial_pipeline([main_config, create_config], seed=0)
