from easydict import EasyDict
from ding.entry import serial_pipeline_gail
from hopper_sac_default_config import hopper_sac_default_config, hopper_sac_default_create_config

obs_shape = 11
act_shape = 3
hopper_sac_gail_default_config = dict(
    exp_name='hopper_sac_gail',
    env=dict(
        env_id='Hopper-v3',
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        collector_env_num=1,
        evaluator_env_num=8,
        use_act_scale=True,
        n_evaluator_episode=8,
        stop_value=6000,
    ),
    reward_model=dict(
        type='gail',
        input_size=obs_shape + act_shape,
        hidden_size=256,
        batch_size=64,
        learning_rate=1e-3,
        update_per_collect=100,
        expert_data_path='hopper_sac/expert_data.pkl',
        load_path='hopper_sac/reward_model/ckpt/ckpt_best.pth.tar',  # state_dict of the reward model
        expert_load_path='hopper_sac/ckpt/ckpt_best.pth.tar',  # path to the expert state_dict
        collect_count=100000,
    ),
    policy=dict(
        cuda=True,
        random_collect_size=25000,
        model=dict(
            obs_shape=obs_shape,
            action_shape=act_shape,
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
            ignore_done=False,
            target_theta=0.005,
            discount_factor=0.99,
            alpha=0.2,
            reparameterization=True,
            auto_alpha=False,
        ),
        collect=dict(
            n_sample=64,
            unroll_len=1,
        ),
        command=dict(),
        eval=dict(),
        other=dict(replay_buffer=dict(replay_buffer_size=1000000, ), ),
    ),
)

hopper_sac_gail_default_config = EasyDict(hopper_sac_gail_default_config)
main_config = hopper_sac_gail_default_config

hopper_sac_gail_default_create_config = dict(
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
)
hopper_sac_gail_default_create_config = EasyDict(hopper_sac_gail_default_create_config)
create_config = hopper_sac_gail_default_create_config

if __name__ == "__main__":
    serial_pipeline_gail(
        [main_config, create_config], [hopper_sac_default_config, hopper_sac_default_create_config],
        max_iterations=1000000,
        seed=0,
        collect_data=True
    )
