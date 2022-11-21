from easydict import EasyDict

from ding.entry import serial_pipeline_dream

# environment hypo
env_id = 'Walker2d-v2'
obs_shape = 17
action_shape = 6

# gpu
cuda = True

main_config = dict(
    exp_name='walker2d_mbsac_mbpo_seed0',
    env=dict(
        env_id=env_id,
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        collector_env_num=4,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=100000,
    ),
    policy=dict(
        cuda=cuda,
        # it is better to put random_collect_size in policy.other
        random_collect_size=10000,
        model=dict(
            obs_shape=obs_shape,
            action_shape=action_shape,
            twin_critic=True,
            action_space='reparameterization',
            actor_head_hidden_size=256,
            critic_head_hidden_size=256,
        ),
        learn=dict(
            lambda_=0.8,
            sample_state=False,
            update_per_collect=20,
            batch_size=512,
            learning_rate_q=3e-4,
            learning_rate_policy=3e-4,
            learning_rate_alpha=3e-4,
            ignore_done=False,
            target_theta=0.005,
            discount_factor=0.99,
            alpha=0.2,
            reparameterization=True,
            auto_alpha=False,
        ),
        collect=dict(
            n_sample=8,
            unroll_len=1,
        ),
        command=dict(),
        eval=dict(evaluator=dict(eval_freq=500, )),  # w.r.t envstep
        other=dict(
            # environment buffer
            replay_buffer=dict(replay_buffer_size=1000000, periodic_thruput_seconds=60),
        ),
    ),
    world_model=dict(
        eval_freq=250,  # w.r.t envstep
        train_freq=250,  # w.r.t envstep
        cuda=cuda,
        rollout_length_scheduler=dict(
            type='linear',
            rollout_start_step=30000,
            rollout_end_step=100000,
            rollout_length_min=1,
            rollout_length_max=3,
        ),
        model=dict(
            ensemble_size=7,
            elite_size=5,
            state_size=obs_shape,  # has to be specified
            action_size=action_shape,  # has to be specified
            reward_size=1,
            hidden_size=200,
            use_decay=True,
            batch_size=512,
            holdout_ratio=0.1,
            max_epochs_since_update=5,
            deterministic_rollout=True,
        ),
    ),
)

main_config = EasyDict(main_config)

create_config = dict(
    env=dict(
        type='mbmujoco',
        import_names=['dizoo.mujoco.envs.mujoco_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='mbsac',
        import_names=['ding.policy.mbpolicy.mbsac'],
    ),
    replay_buffer=dict(type='naive', ),
    world_model=dict(
        type='mbpo',
        import_names=['ding.world_model.mbpo'],
    ),
)
create_config = EasyDict(create_config)

if __name__ == '__main__':
    serial_pipeline_dream((main_config, create_config), seed=0, max_env_step=300000)
