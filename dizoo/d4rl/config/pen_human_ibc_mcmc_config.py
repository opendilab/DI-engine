from easydict import EasyDict

cuda = True
multi_gpu = False

main_config = dict(
    exp_name='pen_human_ibc_mcmc_seed0',
    env=dict(
        env_id='pen-human-v0',
        norm_obs=dict(
            use_norm=True, 
            offline_stats=dict(use_offline_stats=True, ),
        ),
        evaluator_env_num=8,
        n_evaluator_episode=8,
        use_act_scale=True,
        stop_value=1e10,
    ),
    policy=dict(
        cuda=cuda,
        model=dict(
            obs_shape=45,
            action_shape=24,
            stochastic_optim=dict(type='mcmc',)
        ),
        learn=dict(
            multi_gpu=multi_gpu,
            train_epoch=1000,
            batch_size=256,
            optim=dict(learning_rate=1e-5,),
            learner=dict(hook=dict(log_show_after_iter=100)),
        ),
        collect=dict(
            data_type='d4rl',
            data_path=None,
        ),
        eval=dict(evaluator=dict(eval_freq=1000,)),
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
