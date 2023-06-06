from easydict import EasyDict

lunarlander_ppo_config = dict(
    exp_name='lunarlander_gcl_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=8,
        env_id='LunarLander-v2',
        n_evaluator_episode=8,
        stop_value=200,
    ),
    reward_model=dict(
        learning_rate=0.001,
        input_size=9,
        batch_size=32,
        continuous=False,
        update_per_collect=20,
        # Users should add their own data path here. Data path should lead to a file to store data or load the stored data.
        # Absolute path is recommended.
        # In DI-engine, it is usually located in ``exp_name`` directory
        # e.g. 'exp_name/expert_data.pkl'
        expert_data_path='lunarlander_ppo_offpolicy_seed0/expert_data.pkl',
        # Users should add their own model path here. Model path should lead to a model.
        # Absolute path is recommended.
        # In DI-engine, it is ``exp_name/ckpt/ckpt_best.pth.tar``.
        expert_model_path='lunarlander_ppo_offpolicy_seed0/ckpt/ckpt_best.pth.tar',
        collect_count=100000,
    ),
    policy=dict(
        cuda=False,
        action_space='discrete',
        recompute_adv=True,
        model=dict(
            obs_shape=8,
            action_shape=4,
            action_space='discrete',
        ),
        learn=dict(
            update_per_collect=8,
            batch_size=800,
            learning_rate=0.001,
            value_weight=0.5,
            entropy_weight=0.01,
            clip_ratio=0.2,
            adv_norm=True,
        ),
        collect=dict(
            collector_logit=True,
            n_sample=800,
            unroll_len=1,
            discount_factor=0.99,
            gae_lambda=0.95,
        ),
    ),
)
lunarlander_ppo_config = EasyDict(lunarlander_ppo_config)
main_config = lunarlander_ppo_config
lunarlander_ppo_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['dizoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='ppo'),
    reward_model=dict(type='guided_cost'),
)
lunarlander_ppo_create_config = EasyDict(lunarlander_ppo_create_config)
create_config = lunarlander_ppo_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial -c lunarlander_ppo_offpolicy_config.py -s 0`
    from ding.entry import collect_demo_data, serial_pipeline_reward_model_offpolicy
    from dizoo.box2d.lunarlander.config.lunarlander_offppo_config import lunarlander_ppo_offpolicy_config, lunarlander_ppo_offpolicy_create_config

    expert_cfg = (lunarlander_ppo_offpolicy_config, lunarlander_ppo_offpolicy_create_config)
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
