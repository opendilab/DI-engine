from easydict import EasyDict

cuda = True
multi_gpu = False

main_config = dict(
    exp_name='hopper_medium_expert_bc_seed0',
    env=dict(
        env_id='hopper-medium-expert-v0',
        norm_obs=dict(
            use_norm=True, 
            offline_stats=dict(use_offline_stats=True, ),
        ),
        evaluator_env_num=8,
        n_evaluator_episode=8,
        use_act_scale=True,
        stop_value=6000,
    ),
    policy=dict(
        cuda=cuda,
        continuous=True,
        loss_type='mse_loss',
        model=dict(
            obs_shape=11,
            action_shape=3,
            action_space='regression',
            actor_head_hidden_size=512,
            actor_head_layer_num=4,
        ),
        learn=dict(
            multi_gpu=multi_gpu,
            train_epoch=15,
            batch_size=256,
            learning_rate=1e-5,
            learner=dict(hook=dict(log_show_after_iter=1000)),
        ),
        collect=dict(
            data_type='d4rl',
            data_path=None,
        ),
        eval=dict(evaluator=dict(eval_freq=-1,)),
    ),
)
main_config = EasyDict(main_config)
main_config = main_config
create_config = dict(
    env=dict(
        type='d4rl',
        import_names=['dizoo.d4rl.envs.d4rl_env'],
    ),
    env_manager=dict(type='base',),
    policy=dict(
        type='bc',
        import_names=['ding.policy.bc'],
    ),
)
create_config = EasyDict(create_config)
create_config = create_config
