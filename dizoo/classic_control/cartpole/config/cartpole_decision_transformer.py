from easydict import EasyDict
import torch
from copy import deepcopy


cartpole_dt_config = dict(
    exp_name='cartpole_dt',
    env=dict(
        env_name='CartPole-v0',
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=195,
    ),
    policy=dict(
        env_name='CartPole-v0',
        priority=True,
        dataset='medium',  # medium / medium-replay / medium-expert
        rtg_scale=1000,  # normalize returns to go
        max_eval_ep_len=1000,  # max len of one episode
        num_eval_ep=10,  # num of evaluation episodes
        batch_size= 64, # training batch size
        lr=1e-4,
        wt_decay=1e-4,
        warmup_steps=10000,
        max_train_iters=200,
        num_updates_per_iter=100,
        context_len=20,
        n_blocks=3,
        embed_dim=128,
        n_heads =1,
        dropout_p=0.1,
        log_dir='/home/puyuan/DI-engine/dizoo/classic_control/cartpole/dt_log',
        device='cuda',
        max_test_ep_len=200,
        model=dict(
            state_dim=4,
            act_dim=2,
            n_blocks=3,
            h_dim=128,
            context_len=20,
            n_heads =1,
            drop_p=0.1,
            continuous=False,
        ),
        discount_factor=0.999,
        nstep=3,
        learn=dict(
            dataset_path='/home/puyuan/DI-engine/dizoo/classic_control/cartpole/dt_data_cartpole/cartpole/expert_data_10eps.pkl',
            train_epoch=3000,
            batch_size=64,
            learning_rate=0.001,
            target_update_freq=100,
            kappa=1.0,
            min_q_weight=4.0,
        ),
        collect=dict(
            # data_type='hdf5',
            # data_path='./cartpole_generation/expert_demos.hdf5',
            unroll_len=1,
        ),
        eval=dict(evaluator=dict(eval_freq=100, )),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.1,
                decay=10000,
            ), replay_buffer=dict(replay_buffer_size=20000, )
        ),
    ),
)
cartpole_dt_config = EasyDict(cartpole_dt_config)
main_config = cartpole_dt_config
cartpole_dt_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='dt'),
)
cartpole_dt_create_config = EasyDict(cartpole_dt_create_config)
create_config = cartpole_dt_create_config


if __name__ == "__main__":
    from ding.entry import serial_pipeline_dt, collect_demo_data, eval, serial_pipeline
    # from dizoo.classic_control.cartpole.config.cartpole_dt_config import main_config, create_config
    main_config.exp_name = 'cartpole_dt'
    # main_config.policy.collect.data_path = '/home/puyuan/DI-engine/dizoo/classic_control/cartpole/dt_data_cartpole/cartpole/expert_demos.pkl'
    # main_config.policy.collect.data_type = 'naive'
    config = deepcopy([main_config, create_config])
    serial_pipeline_dt(config, seed=0)
