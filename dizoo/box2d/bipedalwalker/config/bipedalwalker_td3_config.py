from easydict import EasyDict

bipedalwalker_td3_config = dict(
    exp_name='bipedalwalker_td3_seed0',
    env=dict(
        env_id='BipedalWalker-v3',
        collector_env_num=1,
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
        cuda=True,
        priority=False,
        model=dict(
            obs_shape=24,
            action_shape=4,
            twin_critic=True,
            actor_head_hidden_size=400,
            critic_head_hidden_size=400,
            action_space='regression',
        ),
        learn=dict(
            update_per_collect=4,
            discount_factor=0.99,
            batch_size=128,
            learning_rate_actor=0.001,
            learning_rate_critic=0.001,
            target_theta=0.005,
            ignore_done=False,
            actor_update_freq=2,
            noise=True,
            noise_sigma=0.2,
            noise_range=dict(
                min=-0.5,
                max=0.5,
            ),
        ),
        collect=dict(
            n_sample=256,
            noise_sigma=0.1,
            collector=dict(collect_print_freq=1000, ),
        ),
        eval=dict(evaluator=dict(eval_freq=100, ), ),
        other=dict(replay_buffer=dict(replay_buffer_size=50000, ), ),
    ),
)
bipedalwalker_td3_config = EasyDict(bipedalwalker_td3_config)
main_config = bipedalwalker_td3_config

bipedalwalker_td3_create_config = dict(
    env=dict(
        type='bipedalwalker',
        import_names=['dizoo.box2d.bipedalwalker.envs.bipedalwalker_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='td3'),
)
bipedalwalker_td3_create_config = EasyDict(bipedalwalker_td3_create_config)
create_config = bipedalwalker_td3_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c bipedalwalker_td3_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline([main_config, create_config], seed=0)
