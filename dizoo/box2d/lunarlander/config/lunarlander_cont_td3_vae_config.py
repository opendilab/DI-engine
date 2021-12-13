from easydict import EasyDict
from ding.entry import serial_pipeline_td3_vae

lunarlander_td3vae_config = dict(
    # exp_name='lunarlander_cont_td3_vae_lad6_wu0_rvuc11_upcv10',
    exp_name='lunarlander_cont_td3_vae_lad6_wu0_rvuc11_upcv20',
    # exp_name='lunarlander_cont_td3_vae_lad6_wu0_rvuc21_upcv40',
    # exp_name='lunarlander_cont_td3_vae_lad6_wu0_rvuc3_upcv4',

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
        cuda=False,
        priority=False,
        # random_collect_size=1280,
        random_collect_size=0,
        original_action_shape=2,
        model=dict(
            obs_shape=8,
            action_shape=6,  # 64,  # latent_action_dim
            twin_critic=True,
            actor_head_type='regression',
        ),
        learn=dict(
            warm_up_update=0,
            # warm_up_update=100,
            rl_vae_update_circle=11,  # train rl 10 iter, vae 1 iter
            # vae_train_times_per_update=1,  # TODO(pu)
            update_per_collect_rl=2,
            update_per_collect_vae=20,

            batch_size=128,
            learning_rate_actor=3e-4,
            learning_rate_critic=3e-4,
            learning_rate_vae=3e-4,
            ignore_done=False,  # TODO(pu)
            actor_update_freq=2,
            noise=True,
            noise_sigma=0.1,
            noise_range=dict(
                min=-0.5,
                max=0.5,
            ),
        ),
        collect=dict(
            # n_sample=48,
            # each_iter_n_sample=128,
            each_iter_n_sample=256,
            noise_sigma=0.1,
            collector=dict(collect_print_freq=1000, ),
        ),
        eval=dict(evaluator=dict(eval_freq=100, ), ),
        other=dict(replay_buffer=dict(replay_buffer_size=20000, ), ),
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