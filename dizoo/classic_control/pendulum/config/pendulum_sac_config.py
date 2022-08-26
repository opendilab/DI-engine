from easydict import EasyDict

pendulum_sac_config = dict(
    exp_name='pendulum_sac_seed0',
    env=dict(
        collector_env_num=10,
        evaluator_env_num=5,
        # (bool) Scale output action into legal range.
        act_scale=True,
        n_evaluator_episode=5,
        stop_value=-250,
    ),
    policy=dict(
        cuda=False,
        priority=False,
        random_collect_size=1000,
        model=dict(
            obs_shape=3,
            action_shape=1,
            twin_critic=True,
            action_space='reparameterization',
            actor_head_hidden_size=128,
            critic_head_hidden_size=128,
        ),
        learn=dict(
            update_per_collect=1,
            batch_size=128,
            learning_rate_q=0.001,
            learning_rate_policy=0.001,
            learning_rate_alpha=0.0003,
            ignore_done=True,
            target_theta=0.005,
            discount_factor=0.99,
            auto_alpha=True,
        ),
        collect=dict(n_sample=10, ),
        eval=dict(evaluator=dict(eval_freq=100, )),
        other=dict(replay_buffer=dict(replay_buffer_size=100000, ), ),
    ),
)
pendulum_sac_config = EasyDict(pendulum_sac_config)
main_config = pendulum_sac_config

pendulum_sac_create_config = dict(
    env=dict(
        type='pendulum',
        import_names=['dizoo.classic_control.pendulum.envs.pendulum_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='sac'),
)
pendulum_sac_create_config = EasyDict(pendulum_sac_create_config)
create_config = pendulum_sac_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c pendulum_sac_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline([main_config, create_config], seed=0)
