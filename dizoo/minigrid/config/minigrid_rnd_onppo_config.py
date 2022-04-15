from easydict import EasyDict

collector_env_num = 8
minigrid_ppo_rnd_config = dict(
    exp_name='minigrid_doorkey8_rnd_onppo_seed0',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        # MiniGrid env id: 'MiniGrid-Empty-8x8-v0', 'MiniGrid-FourRooms-v0','MiniGrid-DoorKey-16x16-v0'
        env_id='MiniGrid-DoorKey-8x8-v0',
        max_step=300,
        stop_value=0.96,
    ),
    reward_model=dict(
        intrinsic_reward_type='add',
        intrinsic_reward_weight=0.001,
        # Specifically for sparse reward env minigrid, in this env,
        # if reach goal, the agent get reward ~1, otherwise 0.
        # We should set the intrinsic_reward_weight approximately equal to the inverse of max_episode_steps.
        learning_rate=5e-4,
        obs_shape=2739,
        batch_size=320,
        update_per_collect=10,
        clear_buffer_per_iters=10,
        obs_norm=True,
        obs_norm_clamp_max=5,
        obs_norm_clamp_min=-5,
    ),
    policy=dict(
        recompute_adv=True,
        cuda=True,
        action_space='discrete',
        model=dict(
            obs_shape=2739,
            action_shape=7,
            action_space='discrete',
            encoder_hidden_size_list=[256, 128, 64, 64],
            critic_head_hidden_size=64,
            actor_head_hidden_size=64,
        ),
        learn=dict(
            epoch_per_collect=10,
            update_per_collect=1,
            batch_size=320,
            learning_rate=3e-4,
            value_weight=0.5,
            entropy_weight=0.001,
            clip_ratio=0.2,
            adv_norm=True,
            value_norm=True,
        ),
        collect=dict(
            collector_env_num=collector_env_num,
            n_sample=3200,
            unroll_len=1,
            discount_factor=0.99,
            gae_lambda=0.95,
        ),
    ),
)
minigrid_ppo_rnd_config = EasyDict(minigrid_ppo_rnd_config)
main_config = minigrid_ppo_rnd_config
minigrid_ppo_rnd_create_config = dict(
    env=dict(
        type='minigrid',
        import_names=['dizoo.minigrid.envs.minigrid_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='ppo'),
    reward_model=dict(type='rnd'),
)
minigrid_ppo_rnd_create_config = EasyDict(minigrid_ppo_rnd_create_config)
create_config = minigrid_ppo_rnd_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c minigrid_rnd_config.py -s 0`
    from ding.entry import serial_pipeline_reward_model_onpolicy
    serial_pipeline_reward_model_onpolicy([main_config, create_config], seed=0)
