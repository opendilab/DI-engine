from easydict import EasyDict

cartpole_offppo_config = dict(
    exp_name='cartpole_offppo_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=195,
    ),
    policy=dict(
        cuda=False,
        model=dict(
            obs_shape=4,
            action_shape=2,
            encoder_hidden_size_list=[64, 64, 128],
            critic_head_hidden_size=128,
            actor_head_hidden_size=128,
            action_space='discrete',
        ),
        learn=dict(
            update_per_collect=6,
            batch_size=64,
            learning_rate=0.001,
            value_weight=0.5,
            entropy_weight=0.01,
            clip_ratio=0.2,
            learner=dict(hook=dict(save_ckpt_after_iter=1000)),
        ),
        collect=dict(
            n_sample=128,
            unroll_len=1,
            discount_factor=0.9,
            gae_lambda=0.95,
        ),
        eval=dict(evaluator=dict(eval_freq=40, )),
        other=dict(replay_buffer=dict(replay_buffer_size=5000))
    ),
)
cartpole_offppo_config = EasyDict(cartpole_offppo_config)
main_config = cartpole_offppo_config
cartpole_offppo_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='ppo_offpolicy'),
)
cartpole_offppo_create_config = EasyDict(cartpole_offppo_create_config)
create_config = cartpole_offppo_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c cartpole_offppo_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline((main_config, create_config), seed=0)
