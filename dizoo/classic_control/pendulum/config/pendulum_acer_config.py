from easydict import EasyDict

pendulum_acer_config = dict(
    exp_name='debug_pendulum_ul50_rbs2e3_seed0',
    seed=0,
    env=dict(
        # collector_env_num=10,
        collector_env_num=1,
        evaluator_env_num=5,
        # (bool) Scale output action into legal range.
        act_scale=True,
        n_evaluator_episode=5,
        stop_value=-150,
    ),
    policy=dict(
        cuda=True,
        priority=False,
        priority_IS_weight=False,
        model=dict(
            obs_shape=3,
            action_shape=1,
            continuous_action_space=True,
            q_value_sample_size=20,  # 5
            noise_ratio=0,  # 0.1,
        ),
        learn=dict(
            # grad_clip_type=None,
            # clip_value=None,
            grad_clip_type='clip_norm',
            clip_value=0.5,
            multi_gpu=False,
            update_per_collect=4,
            batch_size=16,
            unroll_len=50,
            # unroll_len=32,
            # value_weight=0.5,
            entropy_weight=0,  # 0.0001,
            discount_factor=0.99,  # 0.997,#0.9,
            load_path=None,
            c_clip_ratio=5,  # 10, #TODO(pu)
            trust_region=True,
            trust_region_value=1.0,
            learning_rate_actor=0.0005,
            learning_rate_critic=0.0005,
            target_theta=0.005,  # TODO(pu)
        ),
        collect=dict(
            n_sample=16,
            unroll_len=50,
            # unroll_len=32,
            discount_factor=0.99,
            gae_lambda=0.95,
            collector=dict(
                type='sample',
                collect_print_freq=500,

            ),
        ),
        eval=dict(evaluator=dict(eval_freq=200, ), ),
        other=dict(replay_buffer=dict(
            replay_buffer_size=2000,  # 1000, 5000 TODO(pu)
            # replay_buffer_size=10000,  # 1000, 5000 TODO(pu)
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
