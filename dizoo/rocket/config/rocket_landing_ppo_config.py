from easydict import EasyDict

rocket_ppo_config = dict(
    exp_name='rocket_landing_onppo_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=2200,
        task='landing',
        max_steps=800,
        replay_path='rocket_landing_onppo_seed0/video',
    ),
    policy=dict(
        cuda=True,
        action_space='discrete',
        model=dict(
            obs_shape=8,
            action_shape=9,
            action_space='discrete',
            encoder_hidden_size_list=[64, 64, 128],
            critic_head_hidden_size=128,
            actor_head_hidden_size=128,
        ),
        learn=dict(
            epoch_per_collect=10,
            batch_size=64,
            learning_rate=3e-4,
            value_weight=0.5,
            entropy_weight=0.01,
            clip_ratio=0.2,
            adv_norm=True,
            value_norm=True,
            learner=dict(hook=dict(save_ckpt_after_iter=100)),
        ),
        collect=dict(
            n_sample=2048,
            unroll_len=1,
            discount_factor=0.99,
            gae_lambda=0.95,
        ),
        eval=dict(evaluator=dict(eval_freq=1000, ), ),
    ),
)
rocket_ppo_config = EasyDict(rocket_ppo_config)
main_config = rocket_ppo_config
rocket_ppo_create_config = dict(
    env=dict(
        type='rocket',
        import_names=['dizoo.rocket.envs.rocket_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='ppo'),
)
rocket_ppo_create_config = EasyDict(rocket_ppo_create_config)
create_config = rocket_ppo_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial_onpolicy -c rocket_landing_ppo_config.py -s 0`
    from ding.entry import serial_pipeline_onpolicy
    serial_pipeline_onpolicy((main_config, create_config), seed=0)
