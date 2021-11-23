from easydict import EasyDict


pendulum_acer_config = dict(
    seed=0,
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
        on_policy=False,
        priority=False,
        priority_IS_weight=False,
        model=dict(
            obs_shape=3,
            action_shape=1,
            continuous_action_space=True,
            q_value_sample_size= 20,
            noise_ratio= 0.1,
        ),
        learn=dict(
            grad_clip_type=None,
            clip_value=None,
            multi_gpu=False,
            update_per_collect=4,
            batch_size=64,#16,
            value_weight=0.5,
            entropy_weight=0.0001,
            discount_factor=0.997,#0.9,
            lambda_=0.95,
            load_path=None,
            unroll_len=32,
            c_clip_ratio=10,
            trust_region=True,
            trust_region_value=1.0,
            learning_rate_actor=0.0005,
            learning_rate_critic=0.0005,
            target_theta=0.001
        ),
        collect=dict(
            n_sample=16,
            unroll_len=32,
            discount_factor=0.9,
            gae_lambda=0.95,
            collector=dict(
                type='sample',
                collect_print_freq=1000,
            ),
        ),
        eval=dict(evaluator=dict(eval_freq=200, ), ),
        other=dict(replay_buffer=dict(
            replay_buffer_size=10000,#0000,
            max_use=16,
        ), ),
    ),
)
pendulum_acer_config = EasyDict(pendulum_acer_config)
main_config = pendulum_acer_config

pendulum_acer_create_config = dict(
    env=dict(
        type='pendulum',
        import_names=['dizoo.classic_control.pendulum.envs.pendulum_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='acer'),
)
pendulum_acer_create_config = EasyDict(pendulum_acer_create_config)
create_config = pendulum_acer_create_config

from ding.entry import serial_pipeline

if __name__ == "__main__":
    serial_pipeline([pendulum_acer_config, pendulum_acer_create_config], seed=0)