from easydict import EasyDict
import torch
from copy import deepcopy

cartpole_dt_config = dict(
    exp_name='cartpole_dt',
    env=dict(
        env_name='CartPole-v0',
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=195,
    ),
    policy=dict(
        device='cuda',
        stop_value=195,
        env_name='CartPole-v0',
        dataset='medium',  # medium / medium-replay / medium-expert
        rtg_scale=1000,  # normalize returns to go
        max_eval_ep_len=1000,  # max len of one episode
        num_eval_ep=10,  # num of evaluation episodes
        batch_size=64,  # training batch size
        # batch_size= 2, # debug
        lr=1e-4,
        wt_decay=1e-4,
        warmup_steps=10000,
        num_updates_per_iter=100,
        context_len=20,
        n_blocks=3,
        embed_dim=128,
        n_heads=1,
        dropout_p=0.1,
        log_dir='/home/puyuan/DI-engine/dizoo/classic_control/cartpole/dt_log',
        max_test_ep_len=200,
        model=dict(
            state_dim=4,
            act_dim=2,
            n_blocks=3,
            h_dim=128,
            context_len=20,
            n_heads=1,
            drop_p=0.1,
            continuous=False,
        ),
        discount_factor=0.999,
        nstep=3,
        learn=dict(
            dataset_path='/home/puyuan/DI-engine/dizoo/classic_control/cartpole/dt_data/data/expert_data_1000eps.pkl',
            learning_rate=0.001,
            target_update_freq=100,
            kappa=1.0,
            min_q_weight=4.0,
        ),
        collect=dict(unroll_len=1, ),
        eval=dict(evaluator=dict(eval_freq=100, )),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.1,
                decay=int(1e4),
            ),
            replay_buffer=dict(replay_buffer_size=int(2e4), )
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
    main_config.exp_name = 'cartpole_dt'
    config = deepcopy([main_config, create_config])
    serial_pipeline_dt(config, seed=0, max_train_iter=200)
