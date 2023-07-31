from easydict import EasyDict
from copy import deepcopy

hopper_dt_config = dict(
    exp_name='dt_log/atari/Pong/Pong_dt_seed0',
    # exp_name='dt_log/atari/Pong/Pong_dt_seed0',
    env=dict(
        env_id='PongNoFrameskip-v4',
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        collector_env_num=1,
        evaluator_env_num=8,
        use_act_scale=True,
        n_evaluator_episode=8,
        stop_value=20,
        frame_stack=4,
        is_train=False,
        episode_num=10000, # stop in breakout
    ),
    policy=dict(
        num_buffers=50,
        num_steps=500000,
        # num_steps=500,
        data_dir_prefix='d4rl_atari/Pong',
        trajectories_per_buffer=10,
        env_type='atari',
        stop_value=105,
        cuda=True,
        env_name='PongNoFrameskip-v4',
        dataset_name='Pong',
        # rtg_target=20,  # max target return to go
        rtg_target=90,  # max target return to go
        # rtg_scale=10,
        max_eval_ep_len=10000,  # max lenght of one episode
        wt_decay=1e-4,
        clip_grad_norm_p=1.0,
        betas = (0.9, 0.95),
        weight_decay=0.1,
        # warmup_steps=100000,
        warmup_steps=10000,
        context_len=30,
        model=dict(
            state_dim=(4, 84, 84),
            # act_dim=6,
            act_dim=4,
            n_embd=128,
            context_len=30,
            n_heads=8,
            n_layer=6,
            embd_pdrop=0.1,
            resid_pdrop = 0.1,
            attn_pdrop = 0.1,
            continuous=False,
        ),
        learn=dict(
            batch_size=128,
            learning_rate=6e-4,
            target_update_freq=100,
        ),
        collect=dict(
            data_type='d4rl_trajectory',
            # data_path='hopper_medium.hdf5',
            data_path='d4rl_atari/Pong',
            unroll_len=1,
        ),
        eval=dict(evaluator=dict(eval_freq=100, ), ),
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

hopper_dt_config = EasyDict(hopper_dt_config)
main_config = hopper_dt_config
hopper_dt_create_config = dict(
    env=dict(
        type='atari',
        import_names=['dizoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='dt'),
)
hopper_dt_create_config = EasyDict(hopper_dt_create_config)
create_config = hopper_dt_create_config

if __name__ == "__main__":
    from ding.entry import serial_pipeline_dt
    config = deepcopy([main_config, create_config])
    serial_pipeline_dt(config, seed=0, max_train_iter=1000)
