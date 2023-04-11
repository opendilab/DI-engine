from easydict import EasyDict

cuda = False
multi_gpu = False

main_config = dict(
    exp_name='pendulum_ibc_seed0',
    env=dict(
        evaluator_env_num=5,
        act_scale=True,
        n_evaluator_episode=5,
        stop_value=-250,
    ),
    policy=dict(
        cuda=cuda,
        model=dict(
            obs_shape=3,
            action_shape=1,
            stochastic_optim=dict(type='mcmc', cuda=cuda,)
        ),
        learn=dict(
            multi_gpu=multi_gpu,
            train_epoch=15,
            batch_size=256,
            optim=dict(learning_rate=1e-5,),
            learner=dict(hook=dict(log_show_after_iter=1000)),
        ),
        collect=dict(
            data_type='hdf5',
            data_path='./pendulum_sac_data_generation/expert_demos.hdf5',
            collector_logit=False,
        ),
        eval=dict(evaluator=dict(eval_freq=-1,)),
    ),
)
pendulum_ibc_config = EasyDict(main_config)
main_config = pendulum_ibc_config

pendulum_ibc_create_config = dict(
    env=dict(
        type='pendulum',
        import_names=['dizoo.classic_control.pendulum.envs.pendulum_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(
        type='ibc',
        import_names=['ding.policy.ibc'],
    ),
)
pendulum_ibc_create_config = EasyDict(pendulum_ibc_create_config)
create_config = pendulum_ibc_create_config
