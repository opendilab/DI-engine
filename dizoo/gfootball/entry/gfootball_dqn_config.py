from easydict import EasyDict

collector_env_num = 8
evaluator_env_num = 3
# debug
# collector_env_num = 1
# evaluator_env_num = 1  
gfootball_dqn_main_config = dict(
    exp_name='data_gfootball/gfootball_dqn_seed0',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        stop_value=999,
    ),
    policy=dict(
        cuda=True,
        nstep=3,
        discount_factor=0.97,
        model=dict(),
        learn=dict(
            update_per_collect=20,
            batch_size=512,
            learning_rate=0.0001,
            target_update_freq=500,
        ),
        collect=dict(n_sample=256),
        eval=dict(evaluator=dict(eval_freq=5000, n_episode=evaluator_env_num)),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.1,
                decay=int(1e5),
            ),
            replay_buffer=dict(replay_buffer_size=int(1e6), ),
            ),
        ),
)
gfootball_dqn_main_config = EasyDict(gfootball_dqn_main_config)
main_config = gfootball_dqn_main_config

gfootball_dqn_create_config = dict(
    env=dict(
        type='gfootball',
        import_names=['dizoo.gfootball.envs.gfootball_env'],
    ),
    # env_manager=dict(type='subprocess'),
    env_manager=dict(type='base'),
    policy=dict(type='dqn'),
)
gfootball_dqn_create_config = EasyDict(gfootball_dqn_create_config)
create_config = gfootball_dqn_create_config

if __name__ == '__main__':
    # or you can enter `ding -m serial -c gfootball_dqn_config.py -s 0`
    from ding.entry import serial_pipeline
    from dizoo.gfootball.model.iql.iql_network import FootballIQL
    football_iql_model = FootballIQL()
    serial_pipeline((main_config, create_config), model=football_iql_model, seed=0, max_env_step=10e6)
