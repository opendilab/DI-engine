from easydict import EasyDict

main_config = dict(
    exp_name="halfcheetah_medium_dd_seed0",
    env=dict(
        env_id='halfcheetah-medium-v2',
        collector_env_num=1,
        evaluator_env_num=8,
        use_act_scale=True,
        n_evaluator_episode=8,
        returns_scale=400.0,
        termination_penalty=-100,
        max_path_length=1000,
        use_padding=True,
        include_returns=True,
        stop_value=8000,
    ),
    policy=dict(
        cuda=True,
        model=dict(
            model='TemporalUnet',
            model_cfg=dict(
                transition_dim=17,
                dim=128,
                dim_mults=[1, 4, 8],
                returns_condition=True,
                condition_dropout=0.25,
                calc_energy=False,
                kernel_size=5,
            ),
            horizon=100,
            obs_dim=17,
            action_dim=6,
            n_timesteps=200,
            hidden_dim=256,
            returns_condition=True,
            ar_inv=False,
            train_only_inv=False,
            predict_epsilon=True,
            condition_guidance_w=1.2,
            loss_discount=1,
        ),
        normalizer='CDFNormalizer',
        learn=dict(
            data_path=None,
            train_epoch=30000,
            batch_size=32,
            learning_rate=2e-4,
            discount_factor=0.99,
            learner=dict(hook=dict(save_ckpt_after_iter=1000000000, )),
        ),
        collect=dict(data_type='diffuser_traj', ),
        eval=dict(
            evaluator=dict(eval_freq=500, ),
            evaluator_env_num=8,
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
    policy=dict(
        type='dd',
    ),
    replay_buffer=dict(type='naive', ),
)
create_config = EasyDict(create_config)
create_config = create_config
