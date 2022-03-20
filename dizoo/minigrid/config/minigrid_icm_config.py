from easydict import EasyDict

minigrid_ppo_icm_config = dict(
    exp_name='minigrid_icm_offppo_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        # minigrid env id: 'MiniGrid-Empty-8x8-v0', 'MiniGrid-FourRooms-v0','MiniGrid-DoorKey-16x16-v0'
        env_id='MiniGrid-DoorKey-8x8-v0',
        stop_value=0.96,
    ),
    reward_model=dict(
        intrinsic_reward_type='add',
        learning_rate=0.001,
        obs_shape=2739,
        batch_size=320,
        update_per_collect=10,
    ),
    policy=dict(
        cuda=True,
        model=dict(
            obs_shape=2739,
            action_shape=7,
            encoder_hidden_size_list=[256, 128, 64, 64],
        ),
        learn=dict(
            update_per_collect=10,
            batch_size=320,
            learning_rate=0.0003,
            value_weight=0.5,
            entropy_weight=0.001,
            clip_ratio=0.2,
            adv_norm=False,
        ),
        collect=dict(
            n_sample=3200,
            unroll_len=1,
            discount_factor=0.99,
            gae_lambda=0.95,
        ),
    ),
)
minigrid_ppo_icm_config = EasyDict(minigrid_ppo_icm_config)
main_config = minigrid_ppo_icm_config
minigrid_ppo_icm_create_config = dict(
    env=dict(
        type='minigrid',
        import_names=['dizoo.minigrid.envs.minigrid_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='ppo_offpolicy'),
    reward_model=dict(type='icm'),
)
minigrid_ppo_icm_create_config = EasyDict(minigrid_ppo_icm_create_config)
create_config = minigrid_ppo_icm_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c minigrid_icm_config.py -s 0`
    from ding.entry import serial_pipeline_reward_model
    serial_pipeline_reward_model([main_config, create_config], seed=0)