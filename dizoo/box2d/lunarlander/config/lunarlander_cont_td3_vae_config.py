from easydict import EasyDict
from ding.entry import serial_pipeline_td3_vae

lunarlander_td3vae_config = dict(
    # exp_name='lunarlander_cont_td3_vae_wu0_vae5rl5_tvtpc1',
    exp_name='lunarlander_cont_td3_vae_wu0_vae5rl5_tvtpc5',
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
            action_shape=2,  # 64,  # action_latent_shape
            twin_critic=True,
            actor_head_type='regression',
        ),
        learn=dict(
            warm_up_update=0,
            # warm_up_update=100,
            vae_update_freq=10,  # TODO(pu)
            rl_update_freq=10,

            train_vae_times_per_update=5,   # TODO(pu)

            update_per_collect=10,  # train vae 5 times, rl 5 times
            batch_size=128,
            learning_rate_actor=0.001,
            learning_rate_critic=0.001,
            learning_rate_vae=0.001,
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
            # each_iter_n_sample=48,
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