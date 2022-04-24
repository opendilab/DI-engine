from easydict import EasyDict

cartpole_ppo_rnd_config = dict(
    exp_name='cartpole_ppo_rnd_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=195,
    ),
    reward_model=dict(
        intrinsic_reward_type='add',
        intrinsic_reward_weight=None,
        # means the relative weight of RND intrinsic_reward.
        # If intrinsic_reward_weight=None, we will automatically set it based on
        # the absolute value of the difference between max and min extrinsic reward in the sampled mini-batch
        # please refer to rnd_reward_model for details.
        intrinsic_reward_rescale=0.001,
        # means the rescale value of RND intrinsic_reward only used when intrinsic_reward_weight is None
        # please refer to rnd_reward_model for details.
        learning_rate=5e-4,
        obs_shape=4,
        batch_size=32,
        update_per_collect=4,
        obs_norm=True,
        obs_norm_clamp_min=-1,
        obs_norm_clamp_max=1,
        clear_buffer_per_iters=10,
    ),
    policy=dict(
        cuda=True,
        model=dict(
            obs_shape=4,
            action_shape=2,
            encoder_hidden_size_list=[64, 64, 128],
            critic_head_hidden_size=128,
            actor_head_hidden_size=128,
        ),
        learn=dict(
            update_per_collect=6,
            batch_size=64,
            learning_rate=0.001,
            value_weight=0.5,
            entropy_weight=0.01,
            clip_ratio=0.2,
        ),
        collect=dict(
            n_sample=128,
            unroll_len=1,
            discount_factor=0.9,
            gae_lambda=0.95,
        ),
        eval=dict(evaluator=dict(eval_freq=100))
    ),
)
cartpole_ppo_rnd_config = EasyDict(cartpole_ppo_rnd_config)
main_config = cartpole_ppo_rnd_config
cartpole_ppo_rnd_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='ppo_offpolicy'),
    reward_model=dict(type='rnd'),
)
cartpole_ppo_rnd_create_config = EasyDict(cartpole_ppo_rnd_create_config)
create_config = cartpole_ppo_rnd_create_config

if __name__ == "__main__":
    from ding.entry import serial_pipeline_reward_model_onpolicy
    serial_pipeline_reward_model_onpolicy((main_config, create_config), seed=0)
    # you can use the following pipeline to execute pure PPO
    # from ding.entry import serial_pipeline_onpolicy
    # serial_pipeline_onpolicy((main_config, create_config), seed=0)
