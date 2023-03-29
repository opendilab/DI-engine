from easydict import EasyDict

lunarlander_sac_config = dict(
    exp_name='lunarlander_cont_sac_seed0',
    env=dict(
        env_id='LunarLanderContinuous-v2',
        collector_env_num=4,
        evaluator_env_num=8,
        # (bool) Scale output action into legal range.
        act_scale=True,
        n_evaluator_episode=8,
        stop_value=200,
    ),
    policy=dict(
        cuda=False,
        random_collect_size=10000,
        model=dict(
            obs_shape=8,
            action_shape=2,
            twin_critic=True,
            action_space='reparameterization',
        ),
        learn=dict(
            update_per_collect=256,
            batch_size=128,
            learning_rate_q=1e-3,
            learning_rate_policy=3e-4,
            learning_rate_alpha=3e-4,
            auto_alpha=True,
        ),
        collect=dict(n_sample=256, ),
        eval=dict(evaluator=dict(eval_freq=1000, ), ),
        other=dict(replay_buffer=dict(replay_buffer_size=int(1e5), ), ),
    ),
)
lunarlander_sac_config = EasyDict(lunarlander_sac_config)
main_config = lunarlander_sac_config

lunarlander_sac_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['dizoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='sac'),
)
lunarlander_sac_create_config = EasyDict(lunarlander_sac_create_config)
create_config = lunarlander_sac_create_config

if __name__ == '__main__':
    # or you can enter `ding -m serial -c lunarlander_cont_sac_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline([main_config, create_config], seed=0)
