from easydict import EasyDict
from copy import deepcopy

main_config = dict(
    exp_name='walker_params_promptdt_seed0',
    env=dict(
        env_id='walker_params',
        collector_env_num=1,
        evaluator_env_num=8,
        use_act_scale=True,
        n_evaluator_episode=8,
        returns_scale=1.0,
        termination_penalty=-100,
        max_path_length=1000,
        use_padding=True,
        include_returns=True,
        normed=False,
        stop_value=8000,
        horizon=32,
        obs_dim=17,
        action_dim=6,
        test_num=1,
    ),
    dataset=dict(
        data_dir_prefix='/mnt/nfs/share/meta/walker_traj/buffers_walker_param_train',
        rtg_scale=1,
        context_len=1,
        stochastic_prompt=False,
        need_prompt=True,
        test_id=[1],#[5,10,22,31,18,1,12,9,25,38],
        cond=False,
        env_param_path='/mnt/nfs/share/meta/walker/env_walker_param_train_task',
        need_next_obs=False,
    ),
    policy=dict(
        cuda=True,
        stop_value=5000,
        max_len=20,
        max_ep_len=200,
        task_num=3,
        train_num=1,
        obs_dim=17,
        act_dim=6,
        state_mean=None,
        state_std=None,
        no_state_normalize=False,
        no_action_normalize=True,
        need_init_dataprocess=True,
        evaluator_env_num=8,
        rtg_target=5000,  # max target return to go
        max_eval_ep_len=1000,  # max lenght of one episode
        wt_decay=1e-4,
        warmup_steps=10000,
        context_len=20,
        weight_decay=0.1,
        clip_grad_norm_p=0.25,
        model=dict(
            state_dim=17,
            act_dim=6,
            n_blocks=3,
            h_dim=128,
            context_len=20,
            n_heads=1,
            drop_p=0.1,
            continuous=True,
            use_prompt=True,
        ),
        batch_size=32,
        learning_rate=1e-4,
        collect=dict(data_type='meta_traj', ),
        learn=dict(
            data_path=None,
            train_epoch=60000,
            gradient_accumulate_every=2,
            batch_size=32,
            learning_rate=1e-4,
            discount_factor=0.99,
            learner=dict(hook=dict(save_ckpt_after_iter=1000000000, )),
            eval_batch_size=8,
            test_num=1,
        ),
        eval=dict(
            evaluator=dict(
                eval_freq=500, 
                test_env_num=1,
            ),
            test_ret=0.9,
        ),
    ),
)

main_config = EasyDict(main_config)
main_config = main_config

create_config = dict(
    env=dict(
        type='meta',
        import_names=['dizoo.meta_mujoco.envs.meta_env'],
    ),
    env_manager=dict(type='meta_subprocess'),
    policy=dict(
        type='promptdt',
    ),
    replay_buffer=dict(type='naive', ),
)
create_config = EasyDict(create_config)
create_config = create_config