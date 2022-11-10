from easydict import EasyDict

obs_shape = 11
act_shape = 3
hopper_sqil_config = dict(
    exp_name='hopper_sqil_sac_seed0',
    env=dict(
        env_id='Hopper-v3',
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        collector_env_num=1,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=6000,
    ),
    policy=dict(
        cuda=True,
        random_collect_size=10000,
        expert_random_collect_size=10000,
        model=dict(
            obs_shape=obs_shape,
            action_shape=act_shape,
            twin_critic=True,
            action_space='reparameterization',
            actor_head_hidden_size=256,
            critic_head_hidden_size=256,
        ),
        nstep=1,
        discount_factor=0.97,
        learn=dict(
            update_per_collect=1,
            batch_size=64,
            learning_rate_q=1e-3,
            learning_rate_policy=1e-3,
            learning_rate_alpha=3e-4,
            ignore_done=False,
            target_theta=0.005,
            discount_factor=0.99,
            alpha=0.2,
            reparameterization=True,
            auto_alpha=True,
        ),
        collect=dict(
            n_sample=16,
            model_path='model_path_placeholder',
            # Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
        ),
        eval=dict(evaluator=dict(eval_freq=500, )),
        other=dict(replay_buffer=dict(replay_buffer_size=1000000, ), ),
    ),
)

hopper_sqil_config = EasyDict(hopper_sqil_config)
main_config = hopper_sqil_config

hopper_sqil_create_config = dict(
    env=dict(
        type='mujoco',
        import_names=['dizoo.mujoco.envs.mujoco_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='sqil_sac', ),
    replay_buffer=dict(type='naive', ),
)
hopper_sqil_create_config = EasyDict(hopper_sqil_create_config)
create_config = hopper_sqil_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial_sqil -c hopper_sqil_sac_config.py -s 0`
    # then input the config you used to generate your expert model in the path mentioned above
    # e.g. hopper_sac_config.py
    from ding.entry import serial_pipeline_sqil
    from dizoo.mujoco.config.hopper_sac_config import hopper_sac_config, hopper_sac_create_config
    expert_main_config = hopper_sac_config
    expert_create_config = hopper_sac_create_config
    serial_pipeline_sqil(
        [main_config, create_config],
        [expert_main_config, expert_create_config],
        max_env_step=3000000,
        seed=0,
    )
