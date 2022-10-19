from easydict import EasyDict

carry_ppo_config = dict(
    exp_name='evogym_carry_ppo_seed1',
    env=dict(
        env_id='Walker-v0',
        robot='carry_bot',
        robot_dir='./dizoo/evogym/envs',
        collector_env_num=4,
        evaluator_env_num=2,
        n_evaluator_episode=2,
        stop_value=100,
        manager=dict(shared_memory=False, ),
        # The path to save the game replay
        # replay_path='./evogym_carry_ppo_seed0/video',
    ),
    policy=dict(
        cuda=True,
        recompute_adv=True,
        #load_path="./evogym_carry_ppo_seed0/ckpt/ckpt_best.pth.tar",
        model=dict(
            obs_shape=66,
            action_shape=12,
            action_space='continuous',
        ),
        action_space='continuous',
        learn=dict(
            epoch_per_collect=10,
            batch_size=256,
            learning_rate=3e-3,
            value_weight=0.5,
            entropy_weight=0.01,
            clip_ratio=0.2,
            adv_norm=True,
            value_norm=True,
        ),
        collect=dict(
            n_sample=2048,
            unroll_len=1,
            discount_factor=0.99,
            gae_lambda=0.97,
        ),
        eval=dict(evaluator=dict(eval_freq=50, )),
    )
)
carry_ppo_config = EasyDict(carry_ppo_config)
main_config = carry_ppo_config

carry_ppo_create_config = dict(
    env=dict(
        type='evogym',
        import_names=['dizoo.evogym.envs.evogym_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='ppo',
        import_names=['ding.policy.ppo'],
    ),
    replay_buffer=dict(type='naive', ),
)
carry_ppo_create_config = EasyDict(carry_ppo_create_config)
create_config = carry_ppo_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c evogym_carry_ppo_config.py -s 0 --env-step 1e7`
    from ding.entry import serial_pipeline_onpolicy
    serial_pipeline_onpolicy((main_config, create_config), seed=0)
