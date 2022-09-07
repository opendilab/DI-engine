from easydict import EasyDict

n_agent = 5
n_landmark = n_agent
collector_env_num = 8
evaluator_env_num = 8
main_config = dict(
    exp_name='ptz_simple_spread_qtran_seed0',
    env=dict(
        env_family='mpe',
        env_id='simple_spread_v2',
        n_agent=n_agent,
        n_landmark=n_landmark,
        max_cycles=100,
        agent_obs_only=False,
        continuous_actions=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        stop_value=0,
    ),
    policy=dict(
        cuda=True,
        model=dict(
            agent_num=n_agent,
            obs_shape=2 + 2 + n_landmark * 2 + (n_agent - 1) * 2 + (n_agent - 1) * 2,
            global_obs_shape=n_agent * 4 + n_landmark * 2 + n_agent * (n_agent - 1) * 2,
            action_shape=5,
            hidden_size_list=[128],
            embedding_size=64,
            lstm_type='gru',
            dueling=False,
        ),
        learn=dict(
            update_per_collect=100,
            batch_size=32,
            learning_rate=0.0005,
            double_q=True,
            target_update_theta=0.001,
            discount_factor=0.99,
            td_weight=1,
            opt_weight=0.1,
            nopt_min_weight=0.0001,
        ),
        collect=dict(
            n_sample=600,
            unroll_len=16,
            env_num=collector_env_num,
        ),
        eval=dict(env_num=evaluator_env_num, ),
        other=dict(
            eps=dict(
                type='exp',
                start=1.0,
                end=0.05,
                decay=100000,
            ),
            replay_buffer=dict(
                replay_buffer_size=15000,
                # (int) The maximum reuse times of each data
                max_reuse=1e+9,
                max_staleness=1e+9,
            ),
        ),
    ),
)
main_config = EasyDict(main_config)
create_config = dict(
    env=dict(
        import_names=['dizoo.petting_zoo.envs.petting_zoo_simple_spread_env'],
        type='petting_zoo',
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='qtran'),
)
create_config = EasyDict(create_config)

ptz_simple_spread_qtran_config = main_config
ptz_simple_spread_qtran_create_config = create_config

if __name__ == '__main__':
    # or you can enter `ding -m serial -c ptz_simple_spread_qtran_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline((main_config, create_config), seed=0)
