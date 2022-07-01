from easydict import EasyDict

cuda = True

main_config = dict(
    exp_name='hopper_medium_expert_ibc_seed0',
    env=dict(
        env_id='hopper-medium-expert-v0',
        evaluator_env_num=8,
        use_act_scale=True,
        n_evaluator_episode=8,
        stop_value=6000,
    ),
    policy=dict(
        cuda=cuda,
        model=dict(
            obs_shape=11,
            action_shape=3,
            hidden_size=256,
            hidden_layer_num=2,
            stochastic_optim=dict(type='dfo', cuda=cuda,)
        ),
        learn=dict(
            holdout_ratio=0.1,
            train_epoch=300,
            batch_size=256,
            optim=dict(learning_rate=1e-5,),
            learner=dict(hook=dict(log_show_after_iter=1000)),
        ),
        collect=dict(
            normalize_states=True,
            data_type='d4rl',
            data_path=None,
        ),
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
