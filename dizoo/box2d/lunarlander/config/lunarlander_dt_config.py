from easydict import EasyDict
import torch
from copy import deepcopy

lunarlander_dt_config = dict(
    exp_name='data_dt/lunarlander_dt_1000eps_rtgt300_meel1000_seed0_debug',
    env=dict(
        env_id='LunarLander-v2',
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=200,
    ),
    policy=dict(
        stop_value=200,
        state_mean=None,
        state_std=None,
        device='cuda',
        env_name='LunarLander-v2',
        rtg_target=300,  # max target reward_to_go
        rtg_scale=150,
        max_eval_ep_len=1000,  # max len of one episode  # TODO
        wt_decay=1e-4,
        warmup_steps=10000,
        context_len=20,  # TODO
        evaluator_env_num=8,
        log_dir='DI-engine/dizoo/box2d/lunarlander/dt_log_1000eps',
        model=dict(
            state_dim=8,
            act_dim=4,
            n_blocks=3,
            h_dim=128,
            context_len=20,
            n_heads=1,
            drop_p=0.1,
            continuous=False,  # TODO
        ),
        discount_factor=0.999,
        nstep=3,
        learn=dict(
            dataset_path='DI-engine/dizoo/box2d/lunarlander/offline_data/dt_data/dqn_data_1000eps.pkl',  # TODO
            learning_rate=3e-4,
            batch_size=64,  # training batch size
            target_update_freq=100,
        ),
        collect=dict(
            data_type='d4rl_trajectory',
            data_path='DI-engine/dizoo/box2d/lunarlander/offline_data/dt_data/dqn_data_1000eps.pkl',
            unroll_len=1,
        ),
        eval=dict(evaluator=dict(eval_freq=100, )),
    ),
)
lunarlander_dt_config = EasyDict(lunarlander_dt_config)
main_config = lunarlander_dt_config
lunarlander_dt_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['dizoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='dt'),
)
lunarlander_dt_create_config = EasyDict(lunarlander_dt_create_config)
create_config = lunarlander_dt_create_config
