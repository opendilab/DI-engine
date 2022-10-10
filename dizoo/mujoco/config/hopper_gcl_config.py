from easydict import EasyDict

hopper_gcl_config = dict(
    exp_name='hopper_gcl_seed0',
    env=dict(
        env_id='Hopper-v3',
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        collector_env_num=4,
        evaluator_env_num=10,
        n_evaluator_episode=10,
        stop_value=3000,
    ),
    reward_model=dict(
        learning_rate=0.001,
        input_size=14,
        batch_size=32,
        action_shape=3,
        continuous=True,
        update_per_collect=20,
    ),
    policy=dict(
        cuda=False,
        recompute_adv=True,
        action_space='continuous',
        model=dict(
            obs_shape=11,
            action_shape=3,
            action_space='continuous',
        ),
        learn=dict(
            update_per_collect=10,
            batch_size=64,
            learning_rate=3e-4,
            value_weight=0.5,
            entropy_weight=0.0,
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
            n_sample=2048,
            unroll_len=1,
            discount_factor=0.99,
            gae_lambda=0.97,
        ),
        eval=dict(evaluator=dict(eval_freq=100, )),
    ),
)
hopper_gcl_config = EasyDict(hopper_gcl_config)
main_config = hopper_gcl_config

hopper_gcl_create_config = dict(
    env=dict(
        type='mujoco',
        import_names=['dizoo.mujoco.envs.mujoco_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='ppo', ),
    reward_model=dict(type='guided_cost'),
)
hopper_gcl_create_config = EasyDict(hopper_gcl_create_config)
create_config = hopper_gcl_create_config

if __name__ == '__main__':
    from ding.entry import serial_pipeline_guided_cost
    serial_pipeline_guided_cost((main_config, create_config), seed=0)
