from easydict import EasyDict

agent_num = 4
obs_dim = 34
collector_env_num = 8
evaluator_env_num = 32

gfootball_keeper_masac_default_config = dict(
    exp_name='gfootball_counter_masac_seed0',
    env=dict(
        env_name='academy_counterattack_hard',
        agent_num=agent_num,
        obs_dim=obs_dim,
        n_evaluator_episode=32,
        stop_value=1,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        manager=dict(
            shared_memory=False,
            reset_timeout=6000,
        ),
    ),
    policy=dict(
        cuda=True,
        # share_weight=True,
        random_collect_size=int(1e4),
        model=dict(
            agent_num=agent_num,
            agent_obs_shape=34,
            global_obs_shape=68,
            action_shape=19,
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
            unroll_len=1,
        ),
        eval=dict(
            evaluator=dict(eval_freq=50, ),
            env_num=evaluator_env_num,
        ),
        other=dict(
            eps=dict(
                type='linear',
                start=1,
                end=0.05,
                decay=int(5e4),
            ),
            replay_buffer=dict(replay_buffer_size=int(1e6), ),
        ),
    ),
)

gfootball_keeper_masac_default_config = EasyDict(gfootball_keeper_masac_default_config)
main_config = gfootball_keeper_masac_default_config

gfootball_keeper_masac_default_create_config = dict(
    env=dict(
        type='gfootball-academy',
        import_names=['dizoo.gfootball.envs.gfootball_academy_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='discrete_sac'),
)
gfootball_keeper_masac_default_create_config = EasyDict(gfootball_keeper_masac_default_create_config)
create_config = gfootball_keeper_masac_default_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c gfootball_counter_masac_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline([main_config, create_config], seed=0)
