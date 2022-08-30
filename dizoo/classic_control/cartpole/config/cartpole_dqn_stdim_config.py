from easydict import EasyDict

cartpole_dqn_stdim_config = dict(
    exp_name='cartpole_dqn_stdim_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=195,
        replay_path='cartpole_dqn_stdim_seed0/video',
    ),
    policy=dict(
        cuda=False,
        load_path='cartpole_dqn_stdim_seed0/ckpt/ckpt_best.pth.tar',  # necessary for eval
        model=dict(
            obs_shape=4,
            action_shape=2,
            encoder_hidden_size_list=[128, 128, 64],
            dueling=True,
        ),
        aux_model=dict(
            encode_shape=64,
            heads=[1, 1],
            loss_type='infonce',
            temperature=1.0,
        ),
        # the weight of the auxiliary loss to the TD loss
        aux_loss_weight=0.003,
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
cartpole_dqn_stdim_config = EasyDict(cartpole_dqn_stdim_config)
main_config = cartpole_dqn_stdim_config
cartpole_dqn_stdim_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='dqn_stdim'),
    replay_buffer=dict(type='deque', import_names=['ding.data.buffer.deque_buffer_wrapper']),
)
cartpole_dqn_stdim_create_config = EasyDict(cartpole_dqn_stdim_create_config)
create_config = cartpole_dqn_stdim_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c cartpole_dqn_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline((main_config, create_config), seed=0)
