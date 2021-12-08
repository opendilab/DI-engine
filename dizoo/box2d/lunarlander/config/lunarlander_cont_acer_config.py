
from easydict import EasyDict

lunarlander_acer_config = dict(
    exp_name='debug_lunarlander_cont_ul50_bs64_clipnorm0.5_mubound_fixsigma0.3_upc4_ns16_rbs2e3_maxuse16_df0.99_tt0.005_seed0',
    env=dict(
        env_id='LunarLanderContinuous-v2',
        # collector_env_num=10,
        collector_env_num=1,
        evaluator_env_num=5,
        # (bool) Scale output action into legal range.
        act_scale=True,
        n_evaluator_episode=5,
        stop_value=200,
    ),
    policy=dict(
        cuda=True,
        priority=False,
        priority_IS_weight=False,
        model=dict(
            obs_shape=8,
            action_shape=2,
            continuous_action_space=True,
            q_value_sample_size=20,  # 5
            noise_ratio=0,
        ),
        learn=dict(
            # ignore_done=True,
            ignore_done=False,
            grad_clip_type='clip_norm',
            # grad_clip_type='clip_value',
            clip_value=0.5,
            # clip_value=5,
            multi_gpu=False,

            update_per_collect=4,  #TODO(pu)
            # update_per_collect=8,

            batch_size=64,
            unroll_len=50,
            # unroll_len=32,
            entropy_weight=0,  # 0.0001,
            discount_factor=0.99,  # TODO(pu)
            # discount_factor=0.997,

            load_path=None,
            c_clip_ratio=5,  # TODO(pu)
            trust_region=True,
            trust_region_value=1.0,
            learning_rate_actor=0.0005,
            learning_rate_critic=0.0005,
            # target_theta=0.001,
            target_theta=0.005,  # TODO(pu)
            # (float) Weight uniform initialization range in the last output layer
            init_w=3e-3,
            reward_running_norm=False,
            reward_batch_norm=False,
            # reward_running_norm=False,
            # reward_batch_norm=True,
            # reward_running_norm=True,
            # reward_batch_norm=False,
        ),
        collect=dict(
            n_sample=16,
            # n_sample=32,
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
            replay_buffer_size=2000,  # TODO(pu)
            max_use=16,
            # replay_buffer_size=10000,  # TODO(pu)
            # replay_buffer_size=1000,
            # max_use=int(1e4),
        ), ),
    ),
)
lunarlander_acer_config = EasyDict(lunarlander_acer_config)
main_config = lunarlander_acer_config

lunarlander_acer_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['dizoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='acer'),
)
lunarlander_acer_create_config = EasyDict(lunarlander_acer_create_config)
create_config = lunarlander_acer_create_config

from ding.entry import serial_pipeline

if __name__ == "__main__":
    serial_pipeline([lunarlander_acer_config, lunarlander_acer_create_config], seed=0)
