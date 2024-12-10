from easydict import EasyDict

cartpole_dqn_config = dict(
    exp_name='cartpole_dqn_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=195,
        replay_path='cartpole_dqn_seed0/video',
    ),
    policy=dict(
        multi_gpu=True,
        cuda=True,
        model=dict(
            obs_shape=4,
            action_shape=2,
            encoder_hidden_size_list=[128, 128, 64],
            dueling=True,
            # dropout=0.1,
        ),
        nstep=1,
        discount_factor=0.97,
        learn=dict(
            update_per_collect=5,
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
cartpole_dqn_config = EasyDict(cartpole_dqn_config)
main_config = cartpole_dqn_config
cartpole_dqn_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='dqn'),
)
cartpole_dqn_create_config = EasyDict(cartpole_dqn_create_config)
create_config = cartpole_dqn_create_config

if __name__ == "__main__":
    """
    Overview:
        This script should be executed with <nproc_per_node> GPUs.
        Run the following command to launch the script:
        python -m torch.distributed.launch --nproc_per_node=2 --master_port=29501 ./dizoo/classic_control/cartpole/config/cartpole_dqn_ddp_config.py
    """
    from ding.utils import DDPContext
    from ding.entry import serial_pipeline
    with DDPContext():
        serial_pipeline((main_config, create_config), seed=0)

