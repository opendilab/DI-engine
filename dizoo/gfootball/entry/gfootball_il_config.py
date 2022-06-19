from easydict import EasyDict
# debug
# collector_env_num = 1
# evaluator_env_num = 1
collector_env_num = 8
evaluator_env_num = 5
gfootball_il_main_config = dict(
    exp_name='data_gfootball/gfootball_il_seed0',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        stop_value=999,
    ),
    policy=dict(
        cuda=True,
        nstep=1,
        discount_factor=0.97,
        model=dict(),
        learn=dict(
            update_per_collect=20,
            batch_size=512,
            # batch_size=5,  # debug
            learning_rate=0.0001,
            target_update_freq=500,
            learner=dict(load_path=None),
        ),
        collect=dict(n_sample=4096, ),
        eval=dict(evaluator=dict(eval_freq=1000, n_episode=evaluator_env_num)),
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
)
gfootball_il_create_config = EasyDict(gfootball_il_create_config)
create_config = gfootball_il_create_config
