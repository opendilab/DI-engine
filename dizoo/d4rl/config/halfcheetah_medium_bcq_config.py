from easydict import EasyDict

main_config = dict(
    exp_name="halfcheetah_medium_bcq_seed0",
    env=dict(
        env_id='halfcheetah-medium-v2',
        collector_env_num=1,
        evaluator_env_num=8,
        use_act_scale=True,
        n_evaluator_episode=8,
        stop_value=7000,
    ),
    policy=dict(
        cuda=True,
        model=dict(
            obs_shape=17,
            action_shape=6,
            actor_head_hidden_size=[400, 300],
            critic_head_hidden_size=[400, 300],
            phi=0.05,
        ),
        learn=dict(
            data_path=None,
            train_epoch=30000,
            batch_size=100,
            learning_rate_q=3e-3,
            learning_rate_policy=3e-3,
            learning_rate_alpha=3e-3,
            lmbda=0.75,
            learner=dict(hook=dict(save_ckpt_after_iter=1000000000, )),
        ),
        collect=dict(data_type='d4rl', ),
        eval=dict(evaluator=dict(eval_freq=500, )),
        other=dict(replay_buffer=dict(replay_buffer_size=2000000, ), ),
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
        type='bcq',
        import_names=['ding.policy.bcq'],
    ),
    replay_buffer=dict(type='naive', ),
)
create_config = EasyDict(create_config)
create_config = create_config
