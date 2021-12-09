from easydict import EasyDict

bipedalwalker_acer_config = dict(
    exp_name='acer_bipedalwalker_seed0',
    env=dict(
        env_id='bipedalwalker-v3',
        # collector_env_num=10,
        collector_env_num=1,
        evaluator_env_num=5,
        # (bool) Scale output action into legal range.
        act_scale=True,
        n_evaluator_episode=5,
        stop_value=300,
        rew_clip=True,
        replay_path=None,
    ),
    policy=dict(
        cuda=True,
        priority=True,
        n_step=5,
        model=dict(
            obs_shape=24,
            action_shape=4,
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
            unroll_len=32,
            # unroll_len=100,
            # unroll_len=32,
            # value_weight=0.5,
            entropy_weight=0,  # 0.0001,
            discount_factor=0.99,  # 0.997,#0.9,
            load_path=None,
            c_clip_ratio=2,  # 10, #TODO(pu)
            trust_region=True,
            trust_region_value=1.0,
            learning_rate_actor=0.0005,
            learning_rate_critic=0.0005,
            target_theta=0.005,  # TODO(pu)
        ),
        collect=dict(
            n_sample=1024,
            unroll_len=32,
            # unroll_len=100,
            # unroll_len=32,
            discount_factor=0.99,
            gae_lambda=0.95,
            collector=dict(
                # collect_print_freq=1000,
                collect_print_freq=500,

            ),
        ),
        eval=dict(evaluator=dict(eval_freq=200, ), ),
        other=dict(replay_buffer=dict(
            # replay_buffer_size=5000,
            replay_buffer_size=1000,  # TODO(pu)
            max_use=16,
        ), ),
    ),
)
bipedalwalker_acer_config = EasyDict(bipedalwalker_acer_config)
main_config = bipedalwalker_acer_config

bipedalwalker_acer_create_config = dict(
    env=dict(
        type='bipedalwalker',
        import_names=['dizoo.box2d.bipedalwalker.envs.bipedalwalker_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='acer'),
)
bipedalwalker_acer_create_config = EasyDict(bipedalwalker_acer_create_config)
create_config = bipedalwalker_acer_create_config

from ding.entry import serial_pipeline

if __name__ == "__main__":
    serial_pipeline([bipedalwalker_acer_config, bipedalwalker_acer_create_config], seed=0)
