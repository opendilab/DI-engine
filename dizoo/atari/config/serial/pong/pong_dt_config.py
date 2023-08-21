from easydict import EasyDict
from copy import deepcopy

Pong_dt_config = dict(
    exp_name='dt_log/atari/Pong/Pong_dt_seed0',
    env=dict(
        env_id='PongNoFrameskip-v4',
        collector_env_num=1,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=20,
        frame_stack=4,
        is_train=False,
        episode_num=10000,  # stop in breakout
    ),
    dataset=dict(
        env_type='atari',
        num_steps=500000,
        # num_steps=50,
        num_buffers=50,
        rtg_scale=None,
        context_len=30,
        data_dir_prefix='/mnt/nfs/luyd/d4rl_atari/Pong',
        trajectories_per_buffer=10,
    ),
    policy=dict(
        cuda=True,
        multi_gpu=True,
        stop_value=20,
        evaluator_env_num=8,
        rtg_target=20,  # max target return to go
        max_eval_ep_len=10000,  # max lenght of one episode
        wt_decay=1e-4,
        clip_grad_norm_p=1.0,
        weight_decay=0.1,
        warmup_steps=10000,
        model=dict(
            state_dim=(4, 84, 84),
            act_dim=6,
            n_blocks=6,
            h_dim=128,
            context_len=30,
            n_heads=8,
            drop_p=0.1,
            continuous=False,
        ),
        batch_size=128,
        learning_rate=6e-4,
        eval=dict(evaluator=dict(eval_freq=100, ), ),
        collect=dict(
            data_type='d4rl_trajectory',
            unroll_len=1,
        ),
    ),
)

Pong_dt_config = EasyDict(Pong_dt_config)
main_config = Pong_dt_config
Pong_dt_create_config = dict(
    env=dict(
        type='atari',
        import_names=['dizoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='dt'),
)
Pong_dt_create_config = EasyDict(Pong_dt_create_config)
create_config = Pong_dt_create_config
