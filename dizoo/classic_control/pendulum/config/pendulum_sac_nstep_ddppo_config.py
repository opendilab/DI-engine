from easydict import EasyDict

from ding.entry import serial_pipeline_mbrl

# environment hypo
env_id = 'Pendulum-v0'
obs_shape = 3
action_shape = 1

# ddppo
use_gradient_model = True
k = 3
reg = 50

# gpu
cuda = True

# model training hypo
rollout_batch_size = 10000
rollout_retain = 4
rollout_start_step = 2000
rollout_end_step = 15000
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
    exp_name='pendulum_sac_nstep_ddppo',
    env=dict(
        env_id=env_id, # only for backward compatibility
        collector_env_num=10,
        evaluator_env_num=5,
        # (bool) Scale output action into legal range.
        act_scale=True,
        n_evaluator_episode=5,
        stop_value=-250,
    ),
    policy=dict(
        cuda=cuda,
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
            n_step=1,
            n_step_norm=True,
            # 
            update_per_collect=1,
            batch_size=128,
            learning_rate_q=0.001,
            learning_rate_policy=0.001,
            learning_rate_alpha=0.0003,
            ignore_done=True,
            target_theta=0.005,
            discount_factor=0.99,
            auto_alpha=True,
            value_network=False,
        ),
        collect=dict(
            n_sample=1,
            unroll_len=1,
        ),
        command=dict(),
        eval=dict(evaluator=dict(eval_freq=100, )),
        other=dict(replay_buffer=dict(replay_buffer_size=100000, ), ),
    ),
    model_based=dict(
        real_ratio=0.05,
        imagine_buffer=dict(
            type='elastic',
            replay_buffer_size=600000,
            deepcopy=False,
            enable_track_used_data=False,
            set_buffer_size=set_buffer_size,
            periodic_thruput_seconds=60,
        ),
        env_model=dict(
            type='ddppo',
            import_names=['ding.model.template.model_based.ddppo'],
            use_gradient_model=use_gradient_model,
            k=k,
            reg=reg,
            #
            network_size=5,
            elite_size=3,
            state_size=obs_shape,
            action_size=action_shape,
            reward_size=1,
            hidden_size=100,
            use_decay=True,
            batch_size=64,
            holdout_ratio=0.1,
            max_epochs_since_update=5,
            eval_freq=100,
            train_freq=100,
            cuda=cuda,
        ),
        model_env=dict(
            env_id=env_id,
            type='pendulum_model',
            import_names=['dizoo.classic_control.pendulum.envs.pendulum_model_env'],
            rollout_batch_size=rollout_batch_size,
            set_rollout_length=set_rollout_length,
        ),
    ),
)

main_config = EasyDict(main_config)

create_config = dict(
    env=dict(
        type='pendulum',
        import_names=['dizoo.classic_control.pendulum.envs.pendulum_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(
        type='sac_nstep',
        import_names=['ding.policy.sac_nstep'],
    ),
    replay_buffer=dict(type='naive', ),
)
create_config = EasyDict(create_config)


if __name__ == '__main__':
    serial_pipeline_mbrl((main_config, create_config), seed=0)
