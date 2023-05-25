from easydict import EasyDict

main_config = dict(
    exp_name="halfcheetah_medium_expert_edac_seed123",
    env=dict(
        env_id='halfcheetah-medium-expert-v2',
        collector_env_num=1,
        evaluator_env_num=8,
        use_act_scale=True,
        n_evaluator_episode=8,
        stop_value=13000,
    ),
    policy=dict(
        cuda=True,
        model=dict(
            obs_shape=17,
            action_shape=6,
            ensemble_num=10,
            actor_head_hidden_size=256,
            actor_head_layer_num=3,
            critic_head_hidden_size=256,
            critic_head_layer_num=3,
        ),
        learn=dict(
            data_path=None,
            train_epoch=30000,
            batch_size=256,
            learning_rate_q=3e-4,
            learning_rate_policy=3e-4,
            learning_rate_alpha=3e-4,
            alpha=1,
            auto_alpha=True,
            eta=5.0,
            with_q_entropy=False,
            learner=dict(hook=dict(save_ckpt_after_iter=100000, )),
        ),
        collect=dict(data_type='d4rl', ),
        eval=dict(evaluator=dict(eval_freq=500, )),
    ),
    seed=123,
)

main_config = EasyDict(main_config)
main_config = main_config

create_config = dict(
    env=dict(
        type='d4rl',
        import_names=['dizoo.d4rl.envs.d4rl_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(
        type='edac',
        import_names=['ding.policy.edac'],
    ),
)
create_config = EasyDict(create_config)
create_config = create_config
