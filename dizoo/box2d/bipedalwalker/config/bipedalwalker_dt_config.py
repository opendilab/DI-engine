from easydict import EasyDict
from copy import deepcopy

bipedalwalker_dt_config = dict(
    exp_name='bipedalwalker_dt_1000eps_seed0',
    env=dict(
        env_name='BipedalWalker-v3',
        collector_env_num=8,
        evaluator_env_num=5,
        act_scale=True,
        n_evaluator_episode=5,
        stop_value=300,  # stop when return arrive 300
        rew_clip=True,  # reward clip
        replay_path=None,
    ),
    policy=dict(
        stop_value=300,
        device='cuda',
        env_name='BipedalWalker-v3',
        rtg_target=300,  # max target return to go
        max_eval_ep_len=1000,  # max lenght of one episode
        num_eval_ep=10,  # num of evaluation episode
        batch_size=64,
        wt_decay=1e-4,
        warmup_steps=10000,
        num_updates_per_iter=100,
        context_len=20,
        n_blocks=3,
        embed_dim=128,
        n_heads=1,
        dropout_p=0.1,
        log_dir='/home/wangzilin/research/dt/DI-engine/dizoo/box2d/bipedalwalker/dt_data/dt_log_1000eps',
        model=dict(
            state_dim=24,
            act_dim=4,
            n_blocks=3,
            h_dim=128,
            context_len=20,
            n_heads=1,
            drop_p=0.1,
            continuous=True,
        ),
        discount_factor=0.999,
        nstep=3,
        learn=dict(
            dataset_path='/home/wangzilin/research/dt/sac_data_1000eps.pkl',
            learning_rate=0.0001,
            target_update_freq=100,
            kappa=1.0,
            min_q_weight=4.0,
        ),
        collect=dict(unroll_len=1, ),
        eval=dict(evaluator=dict(evalu_freq=100, ), ),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.1,
                decay=10000,
            ),
            replay_buffer=dict(replay_buffer_size=1000, ),
        ),
    ),
)

bipedalwalker_dt_config = EasyDict(bipedalwalker_dt_config)
main_config = bipedalwalker_dt_config
bipedalwalker_dt_create_config = dict(
    env=dict(
        type='bipedalwalker',
        import_names=['dizoo.box2d.bipedalwalker.envs.bipedalwalker_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='dt'),
)
bipedalwalker_dt_create_config = EasyDict(bipedalwalker_dt_create_config)
create_config = bipedalwalker_dt_create_config

if __name__ == "__main__":
    from ding.entry import serial_pipeline_dt
    config = deepcopy([main_config, create_config])
    serial_pipeline_dt(config, seed=0, max_train_iter=1000)
