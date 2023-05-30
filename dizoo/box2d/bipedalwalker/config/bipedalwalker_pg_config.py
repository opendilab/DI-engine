from easydict import EasyDict

bipedalwalker_pg_config = dict(
    exp_name='bipedalwalker_pg_seed0',
    env=dict(
        env_id='BipedalWalker-v3',
        collector_env_num=8,
        evaluator_env_num=8,
        act_scale=True,
        n_evaluator_episode=8,
        stop_value=300,
        rew_clip=True,
    ),
    policy=dict(
        cuda=True,
        action_space='continuous',
        model=dict(
            action_space='continuous',
            obs_shape=24,
            action_shape=4,
        ),
        learn=dict(
            batch_size=64,
            learning_rate=0.001,
            entropy_weight=0.001,
        ),
        collect=dict(
            n_episode=8,
            unroll_len=1,
            discount_factor=0.99,
        ),
        eval=dict(evaluator=dict(eval_freq=200, ))
    ),
)
bipedalwalker_pg_config = EasyDict(bipedalwalker_pg_config)
main_config = bipedalwalker_pg_config
bipedalwalker_pg_create_config = dict(
    env=dict(
        type='bipedalwalker',
        import_names=['dizoo.box2d.bipedalwalker.envs.bipedalwalker_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='pg'),
    collector=dict(type='episode'),
)
bipedalwalker_pg_create_config = EasyDict(bipedalwalker_pg_create_config)
create_config = bipedalwalker_pg_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial_onpolicy -c bipedalwalker_pg_config.py -s 0`
    from ding.entry import serial_pipeline_onpolicy
    serial_pipeline_onpolicy([main_config, create_config], seed=0)
