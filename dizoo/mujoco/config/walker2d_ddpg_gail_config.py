from easydict import EasyDict

walker2d_ddpg_gail_config = dict(
    exp_name='walker2d_ddpg_gail_seed0',
    env=dict(
        env_id='Walker2d-v3',
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        collector_env_num=1,
        evaluator_env_num=8,
        use_act_scale=True,
        n_evaluator_episode=8,
        stop_value=6000,
        manager=dict(shared_memory=False, ),
    ),
    reward_model=dict(
        type='gail',
        input_size=23,
        hidden_size=256,
        batch_size=64,
        learning_rate=1e-3,
        update_per_collect=100,
        # Users should add their own data path here. 
        # Data path should lead to a file to store data or load the stored data.
        # Absolute path is recommended.
        expert_data_path='data_path_placeholder',
        # state_dict of the reward model. Model path should lead to a model.
        # Absolute path is recommended.
        load_path='model_path_placeholder',
        # path to the expert state_dict.
        expert_load_path='model_path_placeholder',
        collect_count=100000,
    ),
    policy=dict(
        # state_dict of the policy.
        # Users should add their own model path here. Model path should lead to a model.
        # Absolute path is recommended.
        # In DI-engine, it is ``exp_name/ckpt/ckpt_best.pth.tar``.
        load_path='walker2d_ddpg_gail/ckpt/ckpt_best.pth.tar',
        cuda=True,
        on_policy=False,
        random_collect_size=25000,
        model=dict(
            obs_shape=17,
            action_shape=6,
            twin_critic=False,
            actor_head_hidden_size=256,
            critic_head_hidden_size=256,
            action_space='regression',
        ),
        learn=dict(
            update_per_collect=1,
            batch_size=256,
            learning_rate_actor=1e-3,
            learning_rate_critic=1e-3,
            ignore_done=False,
            target_theta=0.005,
            discount_factor=0.99,
            actor_update_freq=1,
            noise=False,
        ),
        collect=dict(
            n_sample=64,
            unroll_len=1,
            noise_sigma=0.1,
        ),
        other=dict(replay_buffer=dict(replay_buffer_size=1000000, ), ),
    )
)
walker2d_ddpg_gail_config = EasyDict(walker2d_ddpg_gail_config)
main_config = walker2d_ddpg_gail_config

walker2d_ddpg_gail_create_config = dict(
    env=dict(
        type='mujoco',
        import_names=['dizoo.mujoco.envs.mujoco_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='ddpg',
        import_names=['ding.policy.ddpg'],
    ),
    replay_buffer=dict(type='naive', ),
)
walker2d_ddpg_gail_create_config = EasyDict(walker2d_ddpg_gail_create_config)
create_config = walker2d_ddpg_gail_create_config


if __name__ == "__main__":
    # or you can enter `ding -m serial_gail -c walker2d_ddpg_gail_config.py -s 0`
    from ding.entry import serial_pipeline_gail
    from hopper_sac_config import hopper_sac_config, hopper_sac_create_config
    serial_pipeline_gail(
        [main_config, create_config], [hopper_sac_config, hopper_sac_create_config],
        max_iterations=1000000,
        seed=0,
        collect_data=True
    )
