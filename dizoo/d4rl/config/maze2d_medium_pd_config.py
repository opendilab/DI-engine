from easydict import EasyDict

main_config = dict(
    exp_name="maze2d_medium_pd_seed0",
    env=dict(
        env_id='maze2d-medium-v1',
        collector_env_num=1,
        evaluator_env_num=8,
        use_act_scale=True,
        n_evaluator_episode=8,
        returns_scale=1.0,
        termination_penalty=None,
        max_path_length=40000,
        use_padding=False,
        include_returns=False,
        normed=False,
        stop_value=357,
        horizon=256,
        obs_dim=4,
        action_dim=2,
    ),
    policy=dict(
        cuda=True,
        model=dict(
            diffuser_model='GaussianDiffusion',
            diffuser_model_cfg=dict(
                model='DiffusionUNet1d',
                model_cfg=dict(
                    transition_dim=6,
                    dim=32,
                    dim_mults=[1, 4, 8],
                    returns_condition=False,
                    kernel_size=5,
                    attention=False,
                ),
                horizon=256,
                obs_dim=4,
                action_dim=2,
                n_timesteps=256,
                predict_epsilon=False,
                loss_discount=1,
                clip_denoised=True,
                action_weight=1,
            ),
            value_model=None,
            value_model_cfg=None,
        ),
        normalizer='LimitsNormalizer',
        learn=dict(
            data_path=None,
            train_epoch=60000,
            gradient_accumulate_every=2,
            batch_size=32,
            learning_rate=2e-4,
            discount_factor=0.99,
            plan_batch_size=1,
            include_returns=False,
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
