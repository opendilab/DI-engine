from easydict import EasyDict
import torch
print(torch.cuda.is_available(), torch.__version__)
collector_env_num = 8
evaluator_env_num = 5
cartpole_r2d2_config = dict(
    exp_name='cartpole_r2d2_bs2_n2_ul40_upc4_tuf200_ed1e4_rbs5e3',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=5,
        stop_value=195,
    ),
    policy=dict(
        cuda=False,
        on_policy=False,
        priority=False,
        model=dict(
            obs_shape=4,
            action_shape=2,
            encoder_hidden_size_list=[128, 128, 64],
        ),
        discount_factor=0.997,
        burnin_step=2,#2,
        nstep=2,
        # (int) the trajectory length to unroll the RNN network minus
        # the timestep of burnin operation
        unroll_len=40,#40,
        learn=dict(
            update_per_collect=4,
            batch_size=64,
            learning_rate=0.0005,
            target_update_freq=200,
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
                end=0.1, #0.05,
                decay=10000, 
            ), replay_buffer=dict(replay_buffer_size=5000, ) #5e4
        ),
    ),
)
cartpole_r2d2_config = EasyDict(cartpole_r2d2_config)
main_config = cartpole_r2d2_config
cartpole_r2d2_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='r2d2'),
)
cartpole_r2d2_create_config = EasyDict(cartpole_r2d2_create_config)
create_config = cartpole_r2d2_create_config

if __name__ == "__main__":
    from ding.entry import serial_pipeline
    serial_pipeline([main_config, create_config], seed=0)
