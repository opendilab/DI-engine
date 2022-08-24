from easydict import EasyDict

from ding.entry import serial_pipeline_dream

# environment hypo
env_id = 'Pendulum-v1'
obs_shape = 3
action_shape = 1

# gpu
cuda = False

main_config = dict(
    exp_name='pendulum_mbsac_mbpo_seed0',
    env=dict(
        env_id=env_id,  # only for backward compatibility
        collector_env_num=10,
        evaluator_env_num=5,
        # (bool) Scale output action into legal range.
        act_scale=True,
        n_evaluator_episode=5,
        stop_value=-250,
    ),
    policy=dict(
        cuda=cuda,
        # backward compatibility: it is better to
        # put random_collect_size in policy.other
        random_collect_size=1000,
        model=dict(
            obs_shape=obs_shape,
            action_shape=action_shape,
            twin_critic=True,
            action_space='reparameterization',
            actor_head_hidden_size=128,
            critic_head_hidden_size=128,
        ),
        learn=dict(
            lambda_=0.8,
            sample_state=False,
            update_per_collect=1,
            batch_size=128,
            learning_rate_q=0.001,
            learning_rate_policy=0.001,
            learning_rate_alpha=0.0003,
            ignore_done=True,
            target_theta=0.005,
            discount_factor=0.99,
            auto_alpha=False,
            value_network=False,
        ),
        collect=dict(
            n_sample=1,
            unroll_len=1,
        ),
        command=dict(),
        eval=dict(evaluator=dict(eval_freq=100, )),
        other=dict(
            # environment buffer
            replay_buffer=dict(replay_buffer_size=100000, ),
        ),
    ),
    world_model=dict(
        eval_freq=100,  # w.r.t envstep
        train_freq=100,  # w.r.t envstep
        cuda=cuda,
        rollout_length_scheduler=dict(
            type='linear',
            rollout_start_step=2000,
            rollout_end_step=15000,
            rollout_length_min=3,
            rollout_length_max=3,
        ),
        model=dict(
            ensemble_size=5,
            elite_size=3,
            state_size=obs_shape,
            action_size=action_shape,
            reward_size=1,
            hidden_size=100,
            use_decay=True,
            batch_size=64,
            holdout_ratio=0.1,
            max_epochs_since_update=5,
            deterministic_rollout=True,
        ),
    ),
)

main_config = EasyDict(main_config)

create_config = dict(
    env=dict(
        type='mbpendulum',
        import_names=['dizoo.classic_control.pendulum.envs.pendulum_env'],
    ),
    env_manager=dict(type='base'),
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
    serial_pipeline_dream((main_config, create_config), seed=0)
