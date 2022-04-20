from easydict import EasyDict

cartpole_ppo_icm_config = dict(
    exp_name='cartpole_ppo_icm_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=195,
    ),
    reward_model=dict(
        intrinsic_reward_type='add',
        learning_rate=0.001,
        obs_shape=4,
        action_shape=2,
        batch_size=32,
        update_per_collect=10,
    ),
    policy=dict(
        cuda=False,
        action_space='discrete',
        model=dict(
            obs_shape=4,
            action_shape=2,
            action_space='discrete',
            encoder_hidden_size_list=[64, 64, 128],
            critic_head_hidden_size=128,
            actor_head_hidden_size=128,
        ),
        learn=dict(
            epoch_per_collect=2,
            batch_size=64,
            learning_rate=0.001,
            value_weight=0.5,
            entropy_weight=0.01,
            clip_ratio=0.2,
        ),
        collect=dict(
            n_sample=256,
            unroll_len=1,
            discount_factor=0.9,
            gae_lambda=0.95,
        ),
        eval=dict(evaluator=dict(eval_freq=100, ), ),
    ),
)
cartpole_ppo_icm_config = EasyDict(cartpole_ppo_icm_config)
main_config = cartpole_ppo_icm_config
cartpole_ppo_icm_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='ppo_offpolicy'),
    reward_model=dict(type='icm'),
)
cartpole_ppo_icm_create_config = EasyDict(cartpole_ppo_icm_create_config)
create_config = cartpole_ppo_icm_create_config

if __name__ == '__main__':
    # TODO: confirm which mode to be used in CLI
    from ding.entry import serial_pipeline_reward_model_offpolicy
    serial_pipeline_reward_model_offpolicy([main_config, create_config], seed=0)
