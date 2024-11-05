from easydict import EasyDict

main_config = dict(
    exp_name="halfcheetah_medium_expert_pd_seed0",
    env=dict(
        env_id='halfcheetah-medium-expert-v2',
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
        stop_value=12000,
        horizon=4,
        obs_dim=17,
        action_dim=6,
    ),
    policy=dict(
        cuda=True,
        model=dict(
            diffuser_model='GaussianDiffusion',
            diffuser_model_cfg=dict(
                model='DiffusionUNet1d',
                model_cfg=dict(
                    transition_dim=23,
                    dim=32,
                    dim_mults=[1, 4, 8],
                    returns_condition=False,
                    kernel_size=5,
                    attention=True,
                ),
                horizon=4,
                obs_dim=17,
                action_dim=6,
                n_timesteps=20,
                predict_epsilon=False,
                loss_discount=1,
                action_weight=10,
            ),
            value_model='ValueDiffusion',
            value_model_cfg=dict(
                model='TemporalValue',
                model_cfg=dict(
                    horizon=4,
                    transition_dim=23,
                    dim=32,
                    dim_mults=[1, 4, 8],
                    kernel_size=5,
                ),
                horizon=4,
                obs_dim=17,
                action_dim=6,
                n_timesteps=20,
                predict_epsilon=True,
                loss_discount=1,
            ),
            n_guide_steps=2,
            scale=0.001,
            t_stopgrad=4,
            scale_grad_by_std=True,
        ),
        normalizer='GaussianNormalizer',
        learn=dict(
            data_path=None,
            train_epoch=60000,
            gradient_accumulate_every=2,
            batch_size=32,
            learning_rate=2e-4,
            discount_factor=0.99,
            plan_batch_size=64,
            learner=dict(hook=dict(save_ckpt_after_iter=1000000000, )),
        ),
        collect=dict(data_type='diffuser_traj', ),
        eval=dict(
            evaluator=dict(eval_freq=500, ),
            test_ret=0.9,
        ),
        other=dict(replay_buffer=dict(replay_buffer_size=2000000, ), ),
    ),
)

main_config = EasyDict(main_config)
main_config = main_config

create_config = dict(
    env=dict(
        type='d4rl',
        import_names=['dizoo.d4rl.envs.d4rl_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='pd', ),
    replay_buffer=dict(type='naive', ),
)
create_config = EasyDict(create_config)
create_config = create_config
