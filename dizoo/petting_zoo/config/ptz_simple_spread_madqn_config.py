from easydict import EasyDict

n_agent = 3
n_landmark = n_agent
collector_env_num = 8
evaluator_env_num = 8
main_config = dict(
    exp_name='ptz_simple_spread_madqn_seed0',
    env=dict(
        env_family='mpe',
        env_id='simple_spread_v2',
        n_agent=n_agent,
        n_landmark=n_landmark,
        max_cycles=25,
        agent_obs_only=False,
        agent_specific_global_state=True,
        continuous_actions=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        stop_value=0,
    ),
    policy=dict(
        cuda=True,
        nstep=3,
        model=dict(
            obs_shape=2 + 2 + n_landmark * 2 + (n_agent - 1) * 2 + (n_agent - 1) * 2,
            global_obs_shape=2 + 2 + n_landmark * 2 + (n_agent - 1) * 2 + (n_agent - 1) * 2 + n_agent * (2 + 2) +
            n_landmark * 2 + n_agent * (n_agent - 1) * 2,
            agent_num=n_agent,
            action_shape=5,
            global_cooperation=True,
            hidden_size_list=[256, 256],
        ),
        learn=dict(
            update_per_collect=20,
            batch_size=32,
            learning_rate=0.0005,
            clip_value=5,
            target_update_theta=0.008,
            discount_factor=0.95,
        ),
        collect=dict(
            collector=dict(get_train_sample=True, ),
            n_episode=32,
            unroll_len=10,
            env_num=collector_env_num,
        ),
        command=dict(),
        eval=dict(
            env_num=evaluator_env_num,
            evaluator=dict(eval_freq=1000, ),
        ),
        other=dict(
            eps=dict(
                type='linear',
                start=1,
                end=0.05,
                decay=10000,
            ),
            replay_buffer=dict(replay_buffer_size=15000, ),
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
    policy=dict(type='madqn'),
    collector=dict(type='episode'),
)
create_config = EasyDict(create_config)
ptz_simple_spread_madqn_config = main_config
ptz_simple_spread_madqn_create_config = create_config

if __name__ == '__main__':
    # or you can enter `ding -m serial_entry -c ptz_simple_spread_masac_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline((main_config, create_config), seed=0)
