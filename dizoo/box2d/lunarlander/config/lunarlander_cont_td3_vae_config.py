from easydict import EasyDict
from ding.entry import serial_pipeline_td3_vae

lunarlander_td3vae_config = dict(
    # exp_name='lunarlander_cont_ddpg_vae_lad6_wu10000_ns48_rvuc10_upcr2_upcv0_notargetnoise_collectoriginalnoise_rbs2e4_rsc',# TODO(pu) deubg
    exp_name='lunarlander_cont_ddpg_vae_lad6_wu2e4_ns256_rvuc1_upcr256_upcv100_notargetnoise_collectoriginalnoise_rbs1e6_rsc',# TODO(pu) deubg
    env=dict(
        env_id='LunarLanderContinuous-v2',
        # collector_env_num=8,
        # evaluator_env_num=5,
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
        # random_collect_size=25000,
        random_collect_size=1000,
        original_action_shape=2,
        model=dict(
            obs_shape=8,
            action_shape=6,  # latent_action_dim
            twin_critic=True,
            actor_head_type='regression',
        ),
        learn=dict(
            # warm_up_update=0,
            warm_up_update=int(2e4),
            # vae_train_times_per_update=1,  # TODO(pu)

            rl_vae_update_circle=1,  # train rl 100 iter, vae 1 iter
            update_per_collect_rl=256,
            update_per_collect_vae=100,

            batch_size=256,
            # vae_batch_size=256,
            learning_rate_actor=1e-3,
            learning_rate_critic=1e-3,
            # learning_rate_actor=3e-4,
            # learning_rate_critic=3e-4,
            learning_rate_vae=1e-3,  # 3e-4,
            ignore_done=False,  # TODO(pu)
            target_theta=0.005,
            # actor_update_freq=2,
            actor_update_freq=1,
            # noise=True,
            noise=False,  # TODO(pu)
            noise_sigma=0.1,
            noise_range=dict(
                min=-0.5,
                max=0.5,
            ),
        ),
        collect=dict(
            # each_iter_n_sample=48,  # 1280
            n_sample=256,
            # n_sample=48,
            unroll_len=1,  # TODO(pu)
            # noise_sigma=0.1,
            noise_sigma=0,   # TODO(pu)
            collector=dict(collect_print_freq=1000, ),
        ),
        eval=dict(evaluator=dict(eval_freq=100, ), ),
        # other=dict(replay_buffer=dict(replay_buffer_size=int(2e4), ), ),
        # other=dict(replay_buffer=dict(replay_buffer_size=int(1e5), ), ),
        other=dict(replay_buffer=dict(replay_buffer_size=int(1e6), ), ),

    ),
)
lunarlander_td3vae_config = EasyDict(lunarlander_td3vae_config)
main_config = lunarlander_td3vae_config

lunarlander_td3vae_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['dizoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='td3_vae'),
)
lunarlander_td3vae_create_config = EasyDict(lunarlander_td3vae_create_config)
create_config = lunarlander_td3vae_create_config

if __name__ == '__main__':
    serial_pipeline_td3_vae((main_config, create_config), seed=0)