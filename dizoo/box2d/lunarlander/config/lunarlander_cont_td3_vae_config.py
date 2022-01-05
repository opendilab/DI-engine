from easydict import EasyDict
from ding.entry import serial_pipeline_td3_vae

lunarlander_td3vae_config = dict(
    exp_name=
    'lunarlander_cont_td3_vae_lad6_rcs1e4_wu1e4_ns256_bs128_auf2_targetnoise_collectoriginalnoise_rbs1e5_rsc_lsc_rvuc1_upcr256_upcv10_kw0.01_pw0.01_dot_tanh',
    env=dict(
        env_id='LunarLanderContinuous-v2',
        collector_env_num=8,
        evaluator_env_num=5,
        # (bool) Scale output action into legal range.
        act_scale=True,
        n_evaluator_episode=5,
        stop_value=200,
    ),
    policy=dict(
        cuda=True,
        priority=False,
        random_collect_size=10000,
        original_action_shape=2,
        model=dict(
            obs_shape=8,
            action_shape=6,  # latent_action_dim
            twin_critic=True,
            action_space='regression',
        ),
        learn=dict(
            warm_up_update=int(1e4),
            rl_vae_update_circle=1,  # train rl 1 iter, vae 1 iter
            update_per_collect_rl=256,
            update_per_collect_vae=10,
            batch_size=128,
            learning_rate_actor=3e-4,
            learning_rate_critic=3e-4,
            learning_rate_vae=1e-4,
            ignore_done=False,
            target_theta=0.005,
            actor_update_freq=2,
            noise=True,
            noise_sigma=0.1,
            noise_range=dict(
                min=-0.5,
                max=0.5,
            ),
        ),
        collect=dict(
            n_sample=256,
            unroll_len=1,
            noise_sigma=0,  # NOTE: add noise in original action in _forward_collect method of td3_vae policy
            collector=dict(collect_print_freq=1000, ),
        ),
        eval=dict(evaluator=dict(eval_freq=100, ), ),
        other=dict(replay_buffer=dict(replay_buffer_size=int(1e5), ), ),
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
