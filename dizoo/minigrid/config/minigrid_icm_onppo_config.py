from easydict import EasyDict

collector_env_num = 8
evaluator_env_num = 5
minigrid_icm_onppo_config = dict(
    exp_name='minigrid_AKTDT-7x7_icm_onppo_seed0',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        # minigrid env id: 'MiniGrid-Empty-8x8-v0', 'MiniGrid-FourRooms-v0','MiniGrid-DoorKey-16x16-v0','MiniGrid-AKTDT-7x7-1-v0'
        env_id='MiniGrid-NoisyTV-v0',
        max_step=100,
        stop_value=12,  # run fixed env_steps for MiniGrid-AKTDT-7x7-1-v0
        # stop_value=0.96,
    ),
    reward_model=dict(
        intrinsic_reward_type='add',
        # intrinsic_reward_weight means the relative weight of ICM intrinsic_reward.
        # Specifically for sparse reward env MiniGrid, in this env,
        # if reach goal, the agent get reward ~1, otherwise 0,
        # We could set the intrinsic_reward_weight approximately equal to the inverse of max_episode_steps.
        # Please refer to rnd_reward_model for details.
        intrinsic_reward_weight=0.003,  # 1/300
        learning_rate=3e-4,
        obs_shape=2835,  # 2715 in MiniGrid-AKTDT-7x7-1-v0 env
        batch_size=320,
        update_per_collect=50,
        clear_buffer_per_iters=int(1e3),
        extrinsic_reward_norm=True,
        extrinsic_reward_norm_max=1,
    ),
    policy=dict(
        cuda=True,
        recompute_adv=True,
        action_space='discrete',
        model=dict(
            obs_shape=2835,  # 2715 in MiniGrid-AKTDT-7x7-1-v0 env
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
            n_sample=3200,
            unroll_len=1,
            discount_factor=0.99,
            gae_lambda=0.95,
        ),
        eval=dict(evaluator=dict(eval_freq=1000, )),
    ),
)
minigrid_icm_onppo_config = EasyDict(minigrid_icm_onppo_config)
main_config = minigrid_icm_onppo_config
minigrid_icm_onppo_create_config = dict(
    env=dict(
        type='minigrid',
        import_names=['dizoo.minigrid.envs.minigrid_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='ppo'),
    reward_model=dict(type='icm'),
)
minigrid_icm_onppo_create_config = EasyDict(minigrid_icm_onppo_create_config)
create_config = minigrid_icm_onppo_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c minigrid_icm_onppo_config.py -s 0`
    from ding.entry import serial_pipeline_reward_model_onpolicy
    serial_pipeline_reward_model_onpolicy([main_config, create_config], seed=0, max_env_step=int(10e6))