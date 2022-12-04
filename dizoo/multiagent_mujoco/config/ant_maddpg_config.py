from easydict import EasyDict

ant_ddpg_default_config = dict(
    exp_name='multi_mujoco_ant_2x4_ddpg',
    env=dict(
        scenario='Ant-v2',
        agent_conf="2x4d",
        agent_obsk=2,
        add_agent_id=False,
        episode_limit=1000,
        collector_env_num=8,
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
            action_space='regression',
            actor_head_hidden_size=256,
            critic_head_hidden_size=256,
        ),
        learn=dict(
            update_per_collect=10,
            batch_size=256,
            learning_rate_actor=1e-3,
            learning_rate_critic=1e-3,
            target_theta=0.005,
            discount_factor=0.99,
        ),
        collect=dict(
            n_sample=400,
            noise_sigma=0.1,
        ),
        eval=dict(evaluator=dict(eval_freq=500, )),
        other=dict(replay_buffer=dict(replay_buffer_size=100000, ), ),
    ),
)

ant_ddpg_default_config = EasyDict(ant_ddpg_default_config)
main_config = ant_ddpg_default_config

ant_ddpg_default_create_config = dict(
    env=dict(
        type='mujoco_multi',
        import_names=['dizoo.multiagent_mujoco.envs.multi_mujoco_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='ddpg'),
    replay_buffer=dict(type='naive', ),
)
ant_ddpg_default_create_config = EasyDict(ant_ddpg_default_create_config)
create_config = ant_ddpg_default_create_config

if __name__ == '__main__':
    # or you can enter `ding -m serial -c ant_maddpg_config.py -s 0`
    from ding.entry.serial_entry import serial_pipeline
    serial_pipeline((main_config, create_config), seed=0, max_env_step=int(1e7))
