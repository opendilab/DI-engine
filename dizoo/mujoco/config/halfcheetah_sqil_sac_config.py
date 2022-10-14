from easydict import EasyDict

halfcheetah_sqil_config = dict(
    exp_name='halfcheetah_sqil_sac_seed0',
    env=dict(
        env_id='HalfCheetah-v3',
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        collector_env_num=1,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=12000,
    ),
    policy=dict(
        cuda=True,
        random_collect_size=10000,
        expert_random_collect_size=10000,
        model=dict(
            obs_shape=17,
            action_shape=6,
            twin_critic=True,
            action_space='reparameterization',
            actor_head_hidden_size=256,
            critic_head_hidden_size=256,
        ),
        nstep=1,
        discount_factor=0.97,
        learn=dict(
            update_per_collect=1,
            batch_size=256,
            learning_rate_q=1e-3,
            learning_rate_policy=1e-3,
            learning_rate_alpha=2e-4,
            ignore_done=True,
            target_theta=0.005,
            discount_factor=0.99,
            alpha=0.2,
            reparameterization=True,
            auto_alpha=True,
        ),
        collect=dict(
            n_sample=32,
            # Users should add their own path here (path should lead to a well-trained model)
            model_path='model_path_placeholder',
            # Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
        ),
        eval=dict(evaluator=dict(eval_freq=500, )),  # note: this is the times after which you learns to evaluate
        other=dict(replay_buffer=dict(replay_buffer_size=1000000, ), ),
    ),
)
halfcheetah_sqil_config = EasyDict(halfcheetah_sqil_config)
main_config = halfcheetah_sqil_config
halfcheetah_sqil_create_config = dict(
    env=dict(
        type='mujoco',
        import_names=['dizoo.mujoco.envs.mujoco_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='sqil_sac'),
    replay_buffer=dict(type='naive', ),
)
halfcheetah_sqil_create_config = EasyDict(halfcheetah_sqil_create_config)
create_config = halfcheetah_sqil_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial_sqil -c halfcheetah_sqil_sac_config.py -s 0`
    # then input the config you used to generate your expert model in the path mentioned above
    # e.g. halfcheetah_sac_config.py
    from halfcheetah_sac_config import halfcheetah_sac_config, halfcheetah_sac_create_config
    from ding.entry import serial_pipeline_sqil
    expert_main_config = halfcheetah_sac_config
    expert_create_config = halfcheetah_sac_create_config
    serial_pipeline_sqil(
        [main_config, create_config], [expert_main_config, expert_create_config], seed=0, max_env_step=5000000
    )
