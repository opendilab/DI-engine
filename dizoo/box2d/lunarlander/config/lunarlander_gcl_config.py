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
            # Users should add their own model path here. Model path should lead to a model.
            # Absolute path is recommended.
            # In DI-engine, it is ``exp_name/ckpt/ckpt_best.pth.tar``.
            model_path='model_path_placeholder',
            # If you need the data collected by the collector to contain logit key which reflect the probability of
            # the action, you can change the key to be True.
            # In Guided cost Learning, we need to use logit to train the reward model, we change the key to be True.
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
    from ding.entry import serial_pipeline_guided_cost
    serial_pipeline_guided_cost([main_config, create_config], seed=0)
