from easydict import EasyDict
from ding.entry import serial_pipeline_onpolicy

hopper_ppo_default_config = dict(
    exp_name="hopper_onppo",
    env=dict(
        env_id='Hopper-v3',
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        collector_env_num=8,
        evaluator_env_num=10,
        use_act_scale=True,
        n_evaluator_episode=10,
        stop_value=3000,
    ),
    policy=dict(
        cuda=True,
        recompute_adv=True,
        model=dict(
            obs_shape=11,
            action_shape=3,
            continuous=True,
        ),
        continuous=True,
        learn=dict(
            epoch_per_collect=10,
            update_per_collect=1,
            batch_size=320,
            learning_rate=3e-4,
            value_weight=0.5,
            entropy_weight=0.001,
            clip_ratio=0.2,
            adv_norm=True,
            value_norm=True,
        ),
        collect=dict(
            n_sample=3200,
            unroll_len=1,
            discount_factor=0.99,
            gae_lambda=0.95,
        ),
        eval=dict(evaluator=dict(eval_freq=5000, )),
    ),
)
hopper_ppo_default_config = EasyDict(hopper_ppo_default_config)
main_config = hopper_ppo_default_config

hopper_ppo_create_default_config = dict(
    env=dict(
        type='mujoco',
        import_names=['dizoo.mujoco.envs.mujoco_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='ppo', ),
)
hopper_ppo_create_default_config = EasyDict(hopper_ppo_create_default_config)
create_config = hopper_ppo_create_default_config

if __name__ == "__main__":
    serial_pipeline_onpolicy([main_config, create_config], seed=0)