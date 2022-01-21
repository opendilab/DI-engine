from easydict import EasyDict
from ding.entry import serial_pipeline_guided_cost

halfcheetah_gcl_default_config = dict(
    env=dict(
        env_id='HalfCheetah-v3',
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        collector_env_num=1,
        evaluator_env_num=8,
        use_act_scale=True,
        n_evaluator_episode=8,
        stop_value=12000,
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
        on_policy=False,
        random_collect_size=0,
        model=dict(
            obs_shape=17,
            action_shape=6,
            twin_critic=True,
            action_space='reparameterization',
            actor_head_hidden_size=256,
            critic_head_hidden_size=256,
        ),
        learn=dict(
            update_per_collect=1,
            batch_size=256,
            learning_rate_q=1e-3,
            learning_rate_policy=1e-3,
            learning_rate_alpha=3e-4,
            ignore_done=True,
            target_theta=0.005,
            discount_factor=0.99,
            alpha=0.2,
            reparameterization=True,
            auto_alpha=False,
        ),
        collect=dict(
            demonstration_info_path='path',
            collector_logit=True,
            n_sample=256,
            unroll_len=1,
        ),
        command=dict(),
        eval=dict(),
        other=dict(replay_buffer=dict(replay_buffer_size=1000000, ), ),
    ),
)

halfcheetah_gcl_default_config = EasyDict(halfcheetah_gcl_default_config)
main_config = halfcheetah_gcl_default_config

halfcheetah_gcl_default_create_config = dict(
    env=dict(
        type='mujoco',
        import_names=['dizoo.mujoco.envs.mujoco_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(
        type='sac',
        import_names=['ding.policy.sac'],
    ),
    replay_buffer=dict(type='naive', ),
    reward_model=dict(type='guided_cost'),
)
halfcheetah_gcl_default_create_config = EasyDict(halfcheetah_gcl_default_create_config)
create_config = halfcheetah_gcl_default_create_config

if __name__ == '__main__':
    serial_pipeline_guided_cost((main_config, create_config), seed=0)
