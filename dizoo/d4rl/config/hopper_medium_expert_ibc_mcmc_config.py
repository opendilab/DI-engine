from easydict import EasyDict

cuda = True
multi_gpu = False

main_config = dict(
    exp_name='hopper_medium_expert_ibc_mcmc_seed0',
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
        model=dict(
            obs_shape=11,
            action_shape=3,
            stochastic_optim=dict(type='mcmc',)
        ),
        learn=dict(
            multi_gpu=multi_gpu,
            train_epoch=15,
            batch_size=256,
            optim=dict(learning_rate=1e-5,),
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
        type='ibc',
        import_names=['ding.policy.ibc'],
    ),
)
create_config = EasyDict(create_config)
create_config = create_config
