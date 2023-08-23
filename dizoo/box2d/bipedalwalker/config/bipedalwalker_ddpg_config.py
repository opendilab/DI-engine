from easydict import EasyDict

bipedalwalker_ddpg_config = dict(
    exp_name='bipedalwalker_ddpg_seed0',
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
            twin_critic=False,
            action_space='regression',
            actor_head_hidden_size=400,
            critic_head_hidden_size=400,
        ),
        learn=dict(
            update_per_collect=64,
            batch_size=256,
            learning_rate_actor=0.0003,
            learning_rate_critic=0.0003,
            target_theta=0.005,
            discount_factor=0.99,
            learner=dict(hook=dict(log_show_after_iter=1000, ))
        ),
        collect=dict(n_sample=64, ),
        other=dict(replay_buffer=dict(replay_buffer_size=300000, ), ),
    ),
)
bipedalwalker_ddpg_config = EasyDict(bipedalwalker_ddpg_config)
main_config = bipedalwalker_ddpg_config

bipedalwalker_ddpg_create_config = dict(
    env=dict(
        type='bipedalwalker',
        import_names=['dizoo.box2d.bipedalwalker.envs.bipedalwalker_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='ddpg'),
)
bipedalwalker_ddpg_create_config = EasyDict(bipedalwalker_ddpg_create_config)
create_config = bipedalwalker_ddpg_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c bipedalwalker_ddpg_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline([main_config, create_config], seed=0, max_env_step=int(1e5))
