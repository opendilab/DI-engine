from easydict import EasyDict
from copy import deepcopy

halfcheetah_dt_config = dict(
    exp_name='dt_log/d4rl/halfcheetah/halfcheetah_expert_dt_seed0',
    env=dict(
        env_id='HalfCheetah-v3',
        collector_env_num=1,
        evaluator_env_num=8,
        use_act_scale=True,
        n_evaluator_episode=8,
        stop_value=6000,
    ),
    dataset=dict(
        env_type='mujoco',
        rtg_scale=1000,
        context_len=30,
        data_dir_prefix='d4rl/halfcheetah_expert-v2.pkl',
    ),
    policy=dict(
        cuda=True,
        stop_value=6000,
        state_mean=None,
        state_std=None,
        evaluator_env_num=8,
        env_name='HalfCheetah-v3',
        rtg_target=6000,  # max target return to go
        max_eval_ep_len=1000,  # max lenght of one episode
        wt_decay=1e-4,
        warmup_steps=10000,
        context_len=20,
        weight_decay=0.1,
        clip_grad_norm_p=0.25,
        model=dict(
            state_dim=11,
            act_dim=3,
            n_blocks=3,
            h_dim=128,
            context_len=20,
            n_heads=1,
            drop_p=0.1,
            continuous=True,
        ),
        batch_size=64,
        learning_rate=1e-4,
        collect=dict(
            data_type='d4rl_trajectory',
            unroll_len=1,
        ),
        eval=dict(evaluator=dict(eval_freq=100, ), ),
    ),
)

halfcheetah_dt_config = EasyDict(halfcheetah_dt_config)
main_config = halfcheetah_dt_config
halfcheetah_dt_create_config = dict(
    env=dict(
        type='mujoco',
        import_names=['dizoo.mujoco.envs.mujoco_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='dt'),
)
halfcheetah_dt_create_config = EasyDict(halfcheetah_dt_create_config)
create_config = halfcheetah_dt_create_config

if __name__ == "__main__":
    from ding.entry import serial_pipeline_dt
    config = deepcopy([main_config, create_config])
    serial_pipeline_dt(config, seed=0, max_train_iter=1000)
