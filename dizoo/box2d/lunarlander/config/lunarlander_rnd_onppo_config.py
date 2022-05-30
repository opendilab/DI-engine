from easydict import EasyDict

collector_env_num = 8
evaluator_env_num = 8
lunarlander_ppo_rnd_config = dict(
    exp_name='lunarlander_rnd_onppo_seed0',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        env_id='LunarLander-v2',
        n_evaluator_episode=evaluator_env_num,
        stop_value=200,
    ),
    reward_model=dict(
        intrinsic_reward_type='add',
        # means the relative weight of RND intrinsic_reward.
        # If intrinsic_reward_weight=None, we will automatically set it based on
        # the absolute value of the difference between max and min extrinsic reward in the sampled mini-batch
        # please refer to rnd_reward_model for details.
        intrinsic_reward_weight=None,
        # means the rescale value of RND intrinsic_reward only used when intrinsic_reward_weight is None
        # please refer to rnd_reward_model for details.
        intrinsic_reward_rescale=0.001,
        learning_rate=5e-4,
        obs_shape=8,
        batch_size=320,
        update_per_collect=4,
        obs_norm=True,
        obs_norm_clamp_min=-1,
        obs_norm_clamp_max=1,
        clear_buffer_per_iters=10,
    ),
    policy=dict(
        recompute_adv=True,
        cuda=True,
        action_space='discrete',
        model=dict(
            obs_shape=8,
            action_shape=4,
            action_space='discrete',
        ),
        learn=dict(
            epoch_per_collect=10,
            update_per_collect=1,
            batch_size=64,
            learning_rate=3e-4,
            value_weight=0.5,
            entropy_weight=0.01,
            clip_ratio=0.2,
            adv_norm=True,
            value_norm=True,
        ),
        collect=dict(
            n_sample=512,
            collector_env_num=collector_env_num,
            unroll_len=1,
            discount_factor=0.99,
            gae_lambda=0.95,
        ),
    ),
)
lunarlander_ppo_rnd_config = EasyDict(lunarlander_ppo_rnd_config)
main_config = lunarlander_ppo_rnd_config
lunarlander_ppo_rnd_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['dizoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='ppo'),
    reward_model=dict(type='rnd')
)
lunarlander_ppo_rnd_create_config = EasyDict(lunarlander_ppo_rnd_create_config)
create_config = lunarlander_ppo_rnd_create_config

if __name__ == "__main__":
    from ding.entry import serial_pipeline_reward_model_onpolicy
    serial_pipeline_reward_model_onpolicy([main_config, create_config], seed=0)
