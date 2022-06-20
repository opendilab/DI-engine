from easydict import EasyDict

bipedalwalker_sac_config = dict(
    exp_name='bipedalwalker_sac_seed0',
    env=dict(
        env_id='BipedalWalker-v3',
        collector_env_num=8,
        evaluator_env_num=5,
        # (bool) Scale output action into legal range.
        act_scale=True,
        n_evaluator_episode=5,
        stop_value=300,
        rew_clip=True,
        # The path to save the game replay
        replay_path=None,
    ),
    policy=dict(
        cuda=False,
        priority=False,
        random_collect_size=1000,
        model=dict(
            obs_shape=24,
            action_shape=4,
            twin_critic=True,
            action_space='reparameterization',
            actor_head_hidden_size=128,
            critic_head_hidden_size=128,
        ),
        learn=dict(
            update_per_collect=1,
            batch_size=128,
            learning_rate_q=0.001,
            learning_rate_policy=0.001,
            learning_rate_alpha=0.0003,
            ignore_done=True,
            target_theta=0.005,
            discount_factor=0.99,
            auto_alpha=True,
            value_network=False,
        ),
        collect=dict(
            n_sample=128,
            unroll_len=1,
        ),
        other=dict(replay_buffer=dict(replay_buffer_size=100000, ), ),
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
    policy=dict(
        type='sac',
        import_names=['ding.policy.sac'],
    ),
    replay_buffer=dict(type='naive', ),
)
bipedalwalker_sac_create_config = EasyDict(bipedalwalker_sac_create_config)
create_config = bipedalwalker_sac_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c bipedalwalker_sac_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline([main_config, create_config], seed=0)
