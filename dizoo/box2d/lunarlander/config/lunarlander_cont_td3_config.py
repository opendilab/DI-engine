from easydict import EasyDict
from ding.entry import serial_pipeline

lunarlander_td3_config = dict(
    exp_name='lunarlander_cont_td3_ns256_upcr256_lr3e-4_rbs1e5',
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
        cuda=False,
        priority=False,
        random_collect_size=0,
        model=dict(
            obs_shape=8,
            action_shape=2,
            twin_critic=True,
            action_space='regression',
        ),
        learn=dict(
            update_per_collect=256,
            batch_size=128,
            learning_rate_actor=3e-4,
            learning_rate_critic=3e-4,
            ignore_done=False,
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
            noise_sigma=0.1,
            collector=dict(collect_print_freq=1000, ),
        ),
        eval=dict(evaluator=dict(eval_freq=100, ), ),
        other=dict(replay_buffer=dict(replay_buffer_size=int(1e5), ), ),
    ),
)
lunarlander_td3_config = EasyDict(lunarlander_td3_config)
main_config = lunarlander_td3_config

lunarlander_td3_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['dizoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='td3'),
)
lunarlander_td3_create_config = EasyDict(lunarlander_td3_create_config)
create_config = lunarlander_td3_create_config

if __name__ == '__main__':
    serial_pipeline((main_config, create_config), seed=0)
