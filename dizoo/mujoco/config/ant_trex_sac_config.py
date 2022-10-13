from easydict import EasyDict

ant_trex_sac_config = dict(
    exp_name='ant_trex_sac_seed0',
    env=dict(
        manager=dict(shared_memory=True, reset_inplace=True),
        env_id='Ant-v3',
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        collector_env_num=1,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=6000,
    ),
    reward_model=dict(
        type='trex',
        min_snippet_length=30,
        max_snippet_length=100,
        checkpoint_min=1000,
        checkpoint_max=9000,
        checkpoint_step=1000,
        learning_rate=1e-5,
        update_per_collect=1,
        # Users should add their own model path here. Model path should lead to a model.
        # Absolute path is recommended.
        # In DI-engine, it is ``exp_name/ckpt/ckpt_best.pth.tar``.
        expert_model_path='model_path_placeholder',
        # Path where to store the reward model
        reward_model_path='abs_data_path + ./ant.params',
        continuous=True,
        # Path to the offline dataset
        # See ding/entry/application_entry_trex_collect_data.py to collect the data
        offline_data_path='abs_data_path',
    ),
    policy=dict(
        cuda=True,
        random_collect_size=10000,
        model=dict(
            obs_shape=111,
            action_shape=8,
            twin_critic=True,
            action_space='reparameterization',
            actor_head_hidden_size=256,
            critic_head_hidden_size=256,
        ),
        learn=dict(
            update_per_collect=1,
            batch_size=256,
            learning_rate_q=1e-3,
            learning_rate_policy=1e-3,
            learning_rate_alpha=3e-4,
            ignore_done=False,
            target_theta=0.005,
            discount_factor=0.99,
            alpha=0.2,
            reparameterization=True,
            auto_alpha=False,
        ),
        collect=dict(
            n_sample=1,
            unroll_len=1,
        ),
        command=dict(),
        eval=dict(),
        other=dict(replay_buffer=dict(replay_buffer_size=1000000, ), ),
    ),
)

ant_trex_sac_config = EasyDict(ant_trex_sac_config)
main_config = ant_trex_sac_config

ant_trex_sac_create_config = dict(
    env=dict(
        type='mujoco',
        import_names=['dizoo.mujoco.envs.mujoco_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='sac',
        import_names=['ding.policy.sac'],
    ),
    replay_buffer=dict(type='naive', ),
)
ant_trex_sac_create_config = EasyDict(ant_trex_sac_create_config)
create_config = ant_trex_sac_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c ant_trex_sac_config.py -s 0`
    from ding.entry import serial_pipeline_trex
    serial_pipeline_trex((main_config, create_config), seed=0)
