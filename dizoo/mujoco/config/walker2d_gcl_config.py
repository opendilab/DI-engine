from copy import deepcopy
from ding.entry import serial_pipeline_guided_cost
from easydict import EasyDict

walker_gcl_default_config = dict(
    env=dict(
        env_id='Walker2d-v3',
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        collector_env_num=8,
        evaluator_env_num=10,
        use_act_scale=True,
        n_evaluator_episode=10,
        stop_value=3000,
    ),
    reward_model=dict(
        learning_rate=0.001,
        input_size=23,
        batch_size=32,
        action_shape=6,
        continuous=True,
        update_per_collect=20,
    ),
    policy=dict(
        cuda=False,
        recompute_adv=True,
        action_space='continuous',
        model=dict(
            obs_shape=17,
            action_shape=6,
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
            demonstration_info_path='path',
            n_sample=2048,
            unroll_len=1,
            discount_factor=0.99,
            gae_lambda=0.97,
        ),
        eval=dict(evaluator=dict(eval_freq=100, )),
    ),
)
walker_gcl_default_config = EasyDict(walker_gcl_default_config)
main_config = walker_gcl_default_config

walker_gcl_create_default_config = dict(
    env=dict(
        type='mujoco',
        import_names=['dizoo.mujoco.envs.mujoco_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='ppo', ),
    replay_buffer=dict(type='naive', ),
    reward_model=dict(type='guided_cost'),
)
walker_gcl_create_default_config = EasyDict(walker_gcl_create_default_config)
create_config = walker_gcl_create_default_config

if __name__ == '__main__':
    serial_pipeline_guided_cost((main_config, create_config), seed=0)
