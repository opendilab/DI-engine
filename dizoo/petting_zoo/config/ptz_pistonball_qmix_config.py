from easydict import EasyDict

n_pistons = 20
# n_pistons = 2
collector_env_num = 8
evaluator_env_num = 8

main_config = dict(
    exp_name=f'data_pistonball/ptz_pistonball_n{n_pistons}_qmix_seed0',
    env=dict(
        env_family='butterfly',
        env_id='pistonball_v6',
        n_pistons=n_pistons,
        max_cycles=125,
        agent_obs_only=False,
        continuous_actions=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        stop_value=1e6,
        manager=dict(shared_memory=False,),
        max_env_step=3e6,
    ),
    policy=dict(
        cuda=True,
        model=dict(
            agent_num=n_pistons,
            obs_shape=(3, 457, 120),  # RGB image observation shape for each piston agent
            global_obs_shape=(3, 560, 880),  # Global state shape
            action_shape=3,  # Discrete actions (0, 1, 2)
            hidden_size_list=[128, 128, 64],
            mixer=True,
        ),
        learn=dict(
            update_per_collect=20,
            batch_size=16,
            learning_rate=0.0001,
            target_update_theta=0.001,
            discount_factor=0.99,
            double_q=True,
            clip_value=10,
        ),
        collect=dict(
            n_sample=32,
            unroll_len=16,
            env_num=collector_env_num,
        ),
        eval=dict(env_num=evaluator_env_num),
        other=dict(eps=dict(
            type='exp',
            start=1.0,
            end=0.05,
            decay=100000,
        )),
    ),
)
main_config = EasyDict(main_config)

create_config = dict(
    env=dict(
        import_names=['dizoo.petting_zoo.envs.petting_zoo_pistonball_env'],
        type='petting_zoo_pistonball',
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='qmix'),
)
create_config = EasyDict(create_config)

ptz_pistonball_qmix_config = main_config
ptz_pistonball_qmix_create_config = create_config

if __name__ == '__main__':
    # or you can enter `ding -m serial -c ptz_pistonball_qmix_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline((main_config, create_config), seed=0, max_env_step=main_config.env.max_env_step)