import math

from nervex.loader import dict_, is_type, to_type, collection, interval, is_positive, mcmp, enum, item, raw, check_only

cartpole_ppo_default_loader = dict_(
    env=item('env') >> dict_(
        manager=item('manager') >> dict_(type=item('type') >> enum('base', 'subprocess', 'async_subprocess'), ),
        env_kwargs=item('env_kwargs') >> dict_(
            import_names=item('import_names') >> collection(str),
            env_type=item('env_type') >> is_type(str),
            collector_env_num=item('collector_env_num') >> is_type(int) >> interval(1, 32),
            evaluator_env_num=item('evaluator_env_num') >> is_type(int) >> interval(1, 32),
        ),
    ),
    policy=item('policy') >> dict_(
        use_cuda=item('use_cuda') >> is_type(bool),
        policy_type=item('policy_type') >> is_type(str),
        on_policy=item('on_policy') >> is_type(bool),
        model=item('model') >> dict_(
            obs_dim=item('obs_dim') >> (is_type(int) | collection(int)),
            action_dim=item('action_dim') >> (is_type(int) | collection(int)),
            embedding_dim=item('embedding_dim') >> (is_type(int) | collection(int)),
        ),
        learn=item('learn') >> dict_(
            train_iteration=item('train_iteration') >> is_type(int),
            batch_size=item('batch_size') >> (is_type(int) & interval(1, 128)),
            learning_rate=item('learning_rate') >> interval(0.00001, 0.01),
            weight_decay=item('weight_decay') >> interval(0.0, 0.001),
            algo=item('algo') >> dict_(
                value_weight=item('value_weight') >> (is_type(float) & interval(0, 1)),
                entropy_weight=item('entropy_weight') >> (is_type(float) & interval(0.00, 0.01)),
                clip_ratio=item('clip_ratio') >> (is_type(float) & interval(0.1, 0.3)),
            ),
        ),
        collect=item('collect') >> dict_(
            unroll_len=item('unroll_len') >> is_type(int) >> interval(1, 200),
            algo=item('algo') >> dict_(
                discount_factor=item('discount_factor') >> (is_type(float) & interval(0.9, 0.999)),
                gae_lambda=item('gae_lambda') >> (is_type(float) & interval(0.0, 1.0)),
            )
        ),
    ),
    replay_buffer=item('replay_buffer') >>
    dict_(replay_buffer_size=item('replay_buffer_size') >> is_type(int) >> interval(1, math.inf), ),
    collector=item('collector') >> dict_(
        n_sample=item('n_sample') >> is_type(int) >> interval(8, 128),
        collect_print_freq=item('collect_print_freq') >> is_type(int) >> interval(1, 1000),
    ),
    evaluator=item('evaluator') >> dict_(
        n_episode=item('n_episode') >> is_type(int) >> interval(2, 10),
        eval_freq=item('eval_freq') >> is_type(int) >> interval(1, 500),
        #
        stop_value=item('stop_value') >> is_type(int),
    ),
    learner=item('learner') >> dict_(load_path=item('load_path') >> is_type(str), hook=item('hook')),
)

cartpole_ppo_default_loader = cartpole_ppo_default_loader >> relation_loader

if __name__ == "__main__":
    from cartpole_ppo_default_config import cartpole_ppo_default_config
    print(cartpole_ppo_default_config)
    cartpole_ppo_default_config = cartpole_ppo_default_loader(cartpole_ppo_default_config)
    print(cartpole_ppo_default_config)
