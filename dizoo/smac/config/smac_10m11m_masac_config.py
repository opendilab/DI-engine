from easydict import EasyDict

agent_num = 10
collector_env_num = 8
evaluator_env_num = 8
special_global_state = True

SMAC_10m11m_masac_default_config = dict(
    exp_name='smac_10m11m_masac_seed0',
    env=dict(
        map_name='10m_vs_11m',
        difficulty=7,
        reward_only_positive=True,
        mirror_opponent=False,
        agent_num=agent_num,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=32,
        stop_value=0.99,
        death_mask=False,
        special_global_state=special_global_state,
        manager=dict(
            shared_memory=False,
            reset_timeout=6000,
        ),
    ),
    policy=dict(
        cuda=True,
        multi_agent=True,
        random_collect_size=0,
        model=dict(
            agent_obs_shape=132,
            global_obs_shape=347,
            action_shape=17,
            twin_critic=True,
            actor_head_hidden_size=256,
            critic_head_hidden_size=256,
        ),
        learn=dict(
            update_per_collect=50,
            batch_size=320,
            learning_rate_q=5e-4,
            learning_rate_policy=5e-4,
            learning_rate_alpha=5e-5,
            ignore_done=False,
            target_theta=0.005,
            discount_factor=0.99,
            alpha=0.2,
            auto_alpha=True,
            log_space=True,
        ),
        collect=dict(
            env_num=collector_env_num,
            n_sample=1600,
        ),
        command=dict(),
        eval=dict(
            evaluator=dict(eval_freq=50, ),
            env_num=evaluator_env_num,
        ),
        other=dict(
            eps=dict(
                type='linear',
                start=1,
                end=0.05,
                decay=100000,
            ),
            replay_buffer=dict(replay_buffer_size=50000, ),
        ),
    ),
)

SMAC_10m11m_masac_default_config = EasyDict(SMAC_10m11m_masac_default_config)
main_config = SMAC_10m11m_masac_default_config

SMAC_10m11m_masac_default_create_config = dict(
    env=dict(
        type='smac',
        import_names=['dizoo.smac.envs.smac_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='discrete_sac', ),
)
SMAC_10m11m_masac_default_create_config = EasyDict(SMAC_10m11m_masac_default_create_config)
create_config = SMAC_10m11m_masac_default_create_config

if __name__ == '__main__':

    from ding.entry import serial_pipeline
    serial_pipeline((main_config, create_config), seed=0, max_env_step=int(1e7))
