from easydict import EasyDict

takeoffaviary_ppo_config = dict(
    exp_name='takeoffaviary_ppo_seed0',
    env=dict(
        manager=dict(shared_memory=False, reset_inplace=True),
        env_id='takeoff-aviary-v0',
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        collector_env_num=8,
        evaluator_env_num=8,
        use_act_scale=True,
        n_evaluator_episode=8,
        stop_value=0,
        action_type="VEL",
    ),
    policy=dict(
        cuda=True,
        recompute_adv=True,
        # load_path="./takeoffaviary_ppo_seed0/ckpt/ckpt_best.pth.tar",
        model=dict(
            obs_shape=12,
            action_shape=4,
            action_space='continuous',
        ),
        action_space='continuous',
        learn=dict(
            epoch_per_collect=10,  #reduce
            batch_size=64,
            learning_rate=3e-4,  #tune; pytorch lr scheduler
            value_weight=0.5,
            entropy_weight=0.0,  #0.001
            clip_ratio=0.2,  #0.1
            adv_norm=True,
            value_norm=True,
        ),
        collect=dict(
            n_sample=2048,
            gae_lambda=0.97,
        ),
        eval=dict(evaluator=dict(eval_freq=5000, )),
    ),
)
takeoffaviary_ppo_config = EasyDict(takeoffaviary_ppo_config)
main_config = takeoffaviary_ppo_config

takeoffaviary_ppo_create_config = dict(
    env=dict(
        type='gym_pybullet_drones',
        import_names=['dizoo.gym_pybullet_drones.envs.gym_pybullet_drones_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='ppo', ),
)
takeoffaviary_ppo_create_config = EasyDict(takeoffaviary_ppo_create_config)
create_config = takeoffaviary_ppo_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial_onpolicy -c takeoffaviary_ppo_config.py -s 0 --env-step 1e7`
    from ding.entry import serial_pipeline_onpolicy
    serial_pipeline_onpolicy((main_config, create_config), seed=0)
