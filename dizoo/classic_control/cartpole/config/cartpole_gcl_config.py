from easydict import EasyDict

cartpole_gcl_ppo_offpolicy_config = dict(
    exp_name='cartpole_gcl_offpolicy_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=195,
    ),
    reward_model=dict(
        learning_rate=0.001,
        input_size=5,
        batch_size=32,
        continuous=False,
        expert_data_path='cartpole_ppo_offpolicy_seed0/expert_data.pkl',
        expert_model_path='cartpole_ppo_offpolicy_seed0/ckpt/ckpt_best.pth.tar',
        update_per_collect=10,
        collect_count=1000,
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
cartpole_gcl_ppo_offpolicy_config = EasyDict(cartpole_gcl_ppo_offpolicy_config)
main_config = cartpole_gcl_ppo_offpolicy_config
cartpole_gcl_ppo_offpolicy_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='ppo_offpolicy'),
    reward_model=dict(type='guided_cost'),
)
cartpole_gcl_ppo_offpolicy_create_config = EasyDict(cartpole_gcl_ppo_offpolicy_create_config)
create_config = cartpole_gcl_ppo_offpolicy_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c cartpole_ppo_offpolicy_config.py -s 0`
    from ding.entry import collect_demo_data, serial_pipeline_reward_model_offpolicy
    from dizoo.classic_control.cartpole.config.cartpole_ppo_offpolicy_config import cartpole_ppo_offpolicy_config, cartpole_ppo_offpolicy_create_config

    expert_cfg = (cartpole_ppo_offpolicy_config, cartpole_ppo_offpolicy_create_config)
    expert_data_path = main_config.reward_model.expert_data_path
    state_dict_path = main_config.reward_model.expert_model_path
    collect_count = main_config.reward_model.collect_count
    collect_demo_data(
        expert_cfg,
        seed=0,
        state_dict_path=state_dict_path,
        expert_data_path=expert_data_path,
        collect_count=collect_count
    )
    serial_pipeline_reward_model_offpolicy((main_config, create_config))
