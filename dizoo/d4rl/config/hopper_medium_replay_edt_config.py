from easydict import EasyDict
from copy import deepcopy

hopper_edt_config = dict(
    exp_name='edt_log/d4rl/hopper/hopper_medium_replay_edt_seed0',
    env=dict(
        env_id='Hopper-v3',
        collector_env_num=1,
        evaluator_env_num=8,
        use_act_scale=True,
        n_evaluator_episode=8,
        stop_value=3600,
    ),
    dataset=dict(
        env_type='mujoco',
        rtg_scale=1000,
        context_len=20,
        data_dir_prefix='/d4rl/hopper-medium-replay-v2.pkl',
    ),
    policy=dict(
        cuda=True,
        stop_value=3600,
        state_mean=None,
        state_std=None,
        evaluator_env_num=8,
        env_name='Hopper-v3',
        rtg_target=3600,  # max target return to go
        max_eval_ep_len=1000,  # max lenght of one episode
        wt_decay=1e-4,
        warmup_steps=10000,
        context_len=20,
        weight_decay=0.1,
        clip_grad_norm_p=0.25,
        model=dict(
            state_dim=11,
            act_dim=3,
            n_blocks=4,
            h_dim=512,
            context_len=20,
            n_heads=4,
            drop_p=0.1,
            max_timestep=4096,
            num_bin=60,
            dt_mask=False,
            rtg_scale=1000,
            num_inputs=3,
            real_rtg=False,
            continuous=True,
        ),
        learn=dict(batch_size=128,),
        learning_rate=1e-4,
        collect=dict(
            data_type='d4rl_trajectory',
            unroll_len=1,
        ),
        eval=dict(evaluator=dict(eval_freq=1000, ), ),
    ),
)

hopper_edt_config = EasyDict(hopper_edt_config)
main_config = hopper_edt_config
hopper_edt_create_config = dict(
    env=dict(
        type='mujoco',
        import_names=['dizoo.mujoco.envs.mujoco_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='edt'),
)
hopper_edt_create_config = EasyDict(hopper_edt_create_config)
create_config = hopper_edt_create_config

if __name__ == "__main__":
    from ding.entry import serial_pipeline_edt
    config = deepcopy([main_config, create_config])
    serial_pipeline_edt(config, seed=0, max_train_iter=1000)