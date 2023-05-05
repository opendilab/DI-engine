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
        stop_value=300,
        rew_clip=True,
        # The path to save the game replay
        replay_path=None,
    ),
    policy=dict(
        cuda=False,
        priority=False,
        random_collect_size=1200,
        model=dict(
            obs_shape=24,
            action_shape=4,
            twin_critic=False,
            actor_head_hidden_size=256,
            critic_head_hidden_size=256,
            action_space='regression',
        ),
        learn=dict(
            update_per_collect=1,
            batch_size=128,
            learning_rate_actor=0.001,
            learning_rate_critic=0.001,
            ignore_done=True,
            actor_update_freq=1,
            noise=False,
        ),
        collect=dict(
            n_sample=16,
            noise_sigma=0.1,
            collector=dict(collect_print_freq=1000, ),
        ),
        eval=dict(evaluator=dict(eval_freq=100, )),
        other=dict(replay_buffer=dict(
            replay_buffer_size=20000,
            max_use=16,
        ), ),
    ),
)
bipedalwalker_ddpg_config = EasyDict(bipedalwalker_ddpg_config)
main_config = bipedalwalker_ddpg_config

bipedalwalker_ddpg_create_config = dict(
    env=dict(
        type='bipedalwalker',
        import_names=['dizoo.box2d.bipedalwalker.envs.bipedalwalker_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='ddpg'),
)
bipedalwalker_ddpg_create_config = EasyDict(bipedalwalker_ddpg_create_config)
create_config = bipedalwalker_ddpg_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c bipedalwalker_ddpg_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline([main_config, create_config], seed=0)