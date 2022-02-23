from easydict import EasyDict
from ding.entry.serial_entry import serial_pipeline
ant_sac_default_config = dict(
    exp_name='multi_mujoco_ant_2x4',
    env=dict(
        scenario='Ant-v2',
        agent_conf="2x4d",
        agent_obsk=2,
        add_agent_id=False,
        episode_limit=1000,
        collector_env_num=1,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=6000,
    ),
    policy=dict(
        cuda=True,
        random_collect_size=0,
        multi_agent=True,
        model=dict(
            agent_obs_shape=54,
            global_obs_shape=111,
            action_shape=4,
            twin_critic=True,
            action_space='reparameterization',
            actor_head_hidden_size=256,
            critic_head_hidden_size=256,
        ),
        learn=dict(
            update_per_collect=10,
            batch_size=256,
            learning_rate_q=1e-3,
            learning_rate_policy=1e-3,
            learning_rate_alpha=3e-4,
            ignore_done=False,
            target_theta=0.005,
            discount_factor=0.99,
            alpha=0.2,
            reparameterization=True,
            auto_alpha=True,
            log_space=True,
        ),
        collect=dict(
            n_sample=400,
            unroll_len=1,
        ),
        command=dict(),
        eval=dict(evaluator=dict(eval_freq=100, )),
        other=dict(replay_buffer=dict(replay_buffer_size=100000, ), ),
    ),
)

ant_sac_default_config = EasyDict(ant_sac_default_config)
main_config = ant_sac_default_config

ant_sac_default_create_config = dict(
    env=dict(
        type='mujoco_multi',
        import_names=['dizoo.multiagent_mujoco.envs.multi_mujoco_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(
        type='sac',
        import_names=['ding.policy.sac'],
    ),
    replay_buffer=dict(type='naive', ),
)
ant_sac_default_create_config = EasyDict(ant_sac_default_create_config)
create_config = ant_sac_default_create_config

if __name__ == '__main__':
    serial_pipeline((main_config, create_config), seed=0)
