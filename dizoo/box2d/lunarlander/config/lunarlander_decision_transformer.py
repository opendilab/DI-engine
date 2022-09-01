from easydict import EasyDict
import torch
from copy import deepcopy

lunarlander_dt_config = dict(
    exp_name='data_dt/lunarlander_dt_1000eps_rtgt300_meel1000_seed0_debug',
    env=dict(
        env_name='LunarLander-v2',
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=200,
    ),
    policy=dict(
        stop_value=200,
        device='cuda',
        env_name='LunarLander-v2',
        rtg_target=300,  # max target reward_to_go
        max_eval_ep_len=1000,  # max len of one episode  # TODO
        num_eval_ep=10,  # num of evaluation episodes
        batch_size=64,  # training batch size
        wt_decay=1e-4,
        warmup_steps=10000,
        num_updates_per_iter=100,
        context_len=20,  # TODO
        n_blocks=3,
        embed_dim=128,
        n_heads=1,
        dropout_p=0.1,
        log_dir='/home/puyuan/DI-engine/dizoo/box2d/lunarlander/dt_log_1000eps',
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
            dataset_path='/home/puyuan/DI-engine/dizoo/box2d/lunarlander/dt_data/data/dqn_data_1000eps.pkl',  # TODO
            learning_rate=1e-4,
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
                decay=10000,
            ), replay_buffer=dict(replay_buffer_size=1000, )
        ),
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

if __name__ == "__main__":
    from ding.entry import serial_pipeline_dt, collect_demo_data, eval, serial_pipeline
    config = deepcopy([main_config, create_config])
    serial_pipeline_dt(config, seed=0, max_train_iter=1000)
