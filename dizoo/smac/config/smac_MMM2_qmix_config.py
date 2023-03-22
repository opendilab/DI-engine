from easydict import EasyDict

agent_num = 10
collector_env_num = 16
evaluator_env_num = 8

main_config = dict(
    exp_name='smac_MMM2_qmix_seed0',
    env=dict(
        map_name='MMM2',
        difficulty=7,
        reward_only_positive=True,
        mirror_opponent=False,
        agent_num=agent_num,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        stop_value=0.999,
        n_evaluator_episode=32,
        manager=dict(
            shared_memory=False,
            reset_timeout=6000,
        ),
    ),
    policy=dict(
        model=dict(
            agent_num=agent_num,
            obs_shape=204,
            global_obs_shape=322,
            action_shape=18,
            hidden_size_list=[64],
            mixer=True,
            lstm_type='gru',
            dueling=False,
        ),
        learn=dict(
            update_per_collect=20,
            batch_size=32,
            learning_rate=0.0005,
            clip_value=5,
            double_q=False,
            target_update_theta=0.005,
            discount_factor=0.99,
        ),
        collect=dict(
            n_episode=32,
            unroll_len=10,
            env_num=collector_env_num,
        ),
        eval=dict(env_num=evaluator_env_num, evaluator=dict(eval_freq=100, )),
        other=dict(
            eps=dict(
                type='linear',
                start=1,
                end=0.05,
                decay=10000,
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
        type='smac',
        import_names=['dizoo.smac.envs.smac_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='qmix'),
    collector=dict(type='episode', get_train_sample=True),
)
create_config = EasyDict(create_config)

if __name__ == '__main__':

    from ding.entry import serial_pipeline
    serial_pipeline((main_config, create_config), seed=0)
