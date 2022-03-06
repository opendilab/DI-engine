from easydict import EasyDict
from ding.entry import serial_pipeline_mbrl

# environment hypo
env_id = 'HalfCheetah-v3'
obs_shape = 17
action_shape = 6

# gpu
cuda = True

# model training hypo
rollout_batch_size = 100000
rollout_retain = 4
rollout_start_step = 20000
rollout_end_step = 150000
rollout_length_min = 1
rollout_length_max = 1

x0 = rollout_start_step
y0 = rollout_length_min
y1 = rollout_length_max
w = (rollout_length_max - rollout_length_min) / (rollout_end_step - rollout_start_step)
b = rollout_length_min
set_rollout_length = lambda x: int(min(max(w * (x - x0) + b, y0), y1))
set_buffer_size = lambda x: set_rollout_length(x) * rollout_batch_size * rollout_retain

main_config = dict(
    exp_name='halfcheetach_sac_mbpo_seed0',
    env=dict(
        env_id=env_id,
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        collector_env_num=1,
        evaluator_env_num=8,
        use_act_scale=True,
        n_evaluator_episode=8,
        stop_value=100000,
    ),
    policy=dict(
        cuda=cuda,
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
            update_per_collect=40,
            batch_size=256,
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
            n_sample=1,
            unroll_len=1,
        ),
        command=dict(),
        eval=dict(evaluator=dict(eval_freq=5000, )),
        other=dict(replay_buffer=dict(replay_buffer_size=1000000, periodic_thruput_seconds=60), ),
    ),
    model_based=dict(
        real_ratio=0.05,
        imagine_buffer=dict(
            type='elastic',
            replay_buffer_size=6000000,
            deepcopy=False,
            enable_track_used_data=False,
            set_buffer_size=set_buffer_size,
            periodic_thruput_seconds=60,
        ),
        env_model=dict(
            type='mbpo',
            import_names=['ding.model.template.model_based.mbpo'],
            network_size=7,
            elite_size=5,
            state_size=obs_shape,
            action_size=action_shape,
            reward_size=1,
            hidden_size=200,
            use_decay=True,
            batch_size=256,
            holdout_ratio=0.1,
            max_epochs_since_update=5,
            eval_freq=250,
            train_freq=250,
            cuda=cuda,
        ),
        model_env=dict(
            type='mujoco_model',
            import_names=['dizoo.mujoco.envs.mujoco_model_env'],
            env_id=env_id,
            rollout_batch_size=rollout_batch_size,
            set_rollout_length=set_rollout_length,
        ),
    ),
)

main_config = EasyDict(main_config)

create_config = dict(
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
create_config = EasyDict(create_config)

if __name__ == '__main__':
    serial_pipeline_mbrl((main_config, create_config), seed=0)
