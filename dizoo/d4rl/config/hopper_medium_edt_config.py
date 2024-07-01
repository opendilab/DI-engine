from easydict import EasyDict
from copy import deepcopy

hopper_dt_config = dict(
    exp_name='edt_log/d4rl/hopper/hopper_medium_edt_seed0',
    env=dict(
        env_id='Hopper-v3',
        collector_env_num=1,
        evaluator_env_num=2, 
        use_act_scale=True,
        n_evaluator_episode=2,
        stop_value=3600,
    ),
    dataset=dict(
        env_type='mujoco',
        rtg_scale=1000,
        context_len=20,
        data_dir_prefix='/d4rl/hopper-medium-v2.pkl', #! This points out the directory of dataset
    ),
    policy=dict(
        env_id='Hopper-v3',
        real_rtg=False,
        cuda=True,
        stop_value=3600,
        state_mean=None,
        state_std=None,
        evaluator_env_num=2, #! the evaluator env num in policy should be equal to env
        env_name='Hopper-v3',
        rtg_target=3600,  # max target return to go
        max_eval_ep_len=20,  # max lenght of one episode
        wt_decay=1e-4,
        warmup_steps=10000,
        context_len=20,
        weight_decay=0.1,
        clip_grad_norm_p=0.25,
        model=dict(
            state_dim=11,
            act_dim=3,
            n_blocks=3,
            h_dim=512,
            context_len=20,
            n_heads=1,
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
        weights=dict(
            top_percentile=0.15,
            expectile=0.99,
            expert_weight=10,
            exp_loss_weight=0.5,
            state_loss_weight=1.0,
            cross_entropy_weight=0.001,
            rs_ratio=1, # between 1 and 2
            
        ),
        collect=dict(
            data_type='edt_d4rl_trajectory',
            unroll_len=1,
        ),
        eval=dict(
            evaluator=dict(eval_freq=1000, ), 
            rs_steps=2,
            heuristic=False,
            heuristic_delta=2),
    ),
)

hopper_dt_config = EasyDict(hopper_dt_config)
main_config = hopper_dt_config
hopper_dt_create_config = dict(
    env=dict(
        type='mujoco',
        import_names=['dizoo.mujoco.envs.mujoco_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='edt'),
)
hopper_dt_create_config = EasyDict(hopper_dt_create_config)
create_config = hopper_dt_create_config

if __name__ == "__main__":
    from ding.entry import serial_pipeline_edt
    config = deepcopy([main_config, create_config])
    serial_pipeline_edt(config, seed=0, max_train_iter=1000)