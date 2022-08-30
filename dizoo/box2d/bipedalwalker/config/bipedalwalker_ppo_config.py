from easydict import EasyDict

bipedalwalker_ppo_config = dict(
    exp_name='bipedalwalker_ppo_seed0',
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
        # replay_path='./bipedalwalker_ppo_seed0/video',
    ),
    policy=dict(
        cuda=False,
        load_path="./bipedalwalker_ppo_seed0/ckpt/ckpt_best.pth.tar",
        action_space='continuous',
        model=dict(
            action_space='continuous',
            obs_shape=24,
            action_shape=4,
        ),
        learn=dict(
            epoch_per_collect=10,
            batch_size=64,
            learning_rate=0.001,
            value_weight=0.5,
            entropy_weight=0.01,
            clip_ratio=0.2,
            adv_norm=True,
        ),
        collect=dict(
            n_sample=2048,
            unroll_len=1,
            discount_factor=0.99,
            gae_lambda=0.95,
        ),
    ),
)
bipedalwalker_ppo_config = EasyDict(bipedalwalker_ppo_config)
main_config = bipedalwalker_ppo_config
bipedalwalker_ppo_create_config = dict(
    env=dict(
        type='bipedalwalker',
        import_names=['dizoo.box2d.bipedalwalker.envs.bipedalwalker_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='ppo'),
)
bipedalwalker_ppo_create_config = EasyDict(bipedalwalker_ppo_create_config)
create_config = bipedalwalker_ppo_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial_onpolicy -c bipedalwalker_ppo_config.py -s 0`
    from ding.entry import serial_pipeline_onpolicy
    serial_pipeline_onpolicy([main_config, create_config], seed=0)
