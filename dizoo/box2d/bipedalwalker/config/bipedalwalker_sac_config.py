from easydict import EasyDict

bipedalwalker_sac_config = dict(
    exp_name='bipedalwalker_sac_config0',
    env=dict(
        env_id='BipedalWalker-v3',
        collector_env_num=8,
        evaluator_env_num=5,
        # (bool) Scale output action into legal range.
        act_scale=True,
        n_evaluator_episode=5,
        rew_clip=True,
    ),
    policy=dict(
        cuda=True,
        random_collect_size=10000,
        model=dict(
            obs_shape=24,
            action_shape=4,
            twin_critic=True,
            action_space='reparameterization',
            actor_head_hidden_size=128,
            critic_head_hidden_size=128,
        ),
        learn=dict(
            update_per_collect=64,
            batch_size=256,
            learning_rate_q=0.0003,
            learning_rate_policy=0.0003,
            learning_rate_alpha=0.0003,
            target_theta=0.005,
            discount_factor=0.99,
            auto_alpha=True,
            learner=dict(hook=dict(log_show_after_iter=1000, ))
        ),
        collect=dict(n_sample=64, ),
        other=dict(replay_buffer=dict(replay_buffer_size=300000, ), ),
    ),
)
bipedalwalker_sac_config = EasyDict(bipedalwalker_sac_config)
main_config = bipedalwalker_sac_config
bipedalwalker_sac_create_config = dict(
    env=dict(
        type='bipedalwalker',
        import_names=['dizoo.box2d.bipedalwalker.envs.bipedalwalker_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='sac', ),
    replay_buffer=dict(type='naive', ),
)
bipedalwalker_sac_create_config = EasyDict(bipedalwalker_sac_create_config)
create_config = bipedalwalker_sac_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c bipedalwalker_sac_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline([main_config, create_config], seed=0, max_env_step=int(1e5))
