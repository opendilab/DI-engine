from easydict import EasyDict

env_num = 8
gfootball_il_main_config = dict(
    exp_name='gfootball_rule_seed0',
    env=dict(
        collector_env_num=env_num,
        evaluator_env_num=env_num,
        n_evaluator_episode=env_num,
        stop_value=999,
    ),
    policy=dict(
        cuda=True,
        policy_type='IL',
        nstep=1,
        discount_factor=0.97,
        model=dict(),
        learn=dict(
            batch_size=64,
            learning_rate=0.001,
        ),
        collect=dict(n_sample=4096),
        eval=dict(evaluator=dict(eval_freq=1000, n_episode=env_num)),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.1,
                decay=10000,
            ),
            replay_buffer=dict(replay_buffer_size=int(1e6), ),
            ),
        ),
)
gfootball_il_main_config = EasyDict(gfootball_il_main_config)
main_config = gfootball_il_main_config

gfootball_il_create_config = dict(
    env=dict(
        type='gfootball',
        import_names=['dizoo.gfootball.envs.gfootball_env'],
    ),
    # env_manager=dict(type='subprocess'),
    env_manager=dict(type='base'),
    policy=dict(type='bc'),
    replay_buffer=dict(
        type='deque',
        import_names=['ding.data.buffer.deque_buffer_wrapper']
    ),
)
gfootball_il_create_config = EasyDict(gfootball_il_create_config)
create_config = gfootball_il_create_config
