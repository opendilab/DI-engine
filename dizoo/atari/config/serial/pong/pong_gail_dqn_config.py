from easydict import EasyDict

pong_dqn_gail_config = dict(
    exp_name='pong_gail_dqn_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=20,
        env_id='Pong-v4',
        #'ALE/Pong-v5' is available. But special setting is needed after gym make.
        frame_stack=4,
    ),
    reward_model=dict(
        type='gail',
        input_size=[4, 84, 84],
        hidden_size=128,
        batch_size=64,
        learning_rate=1e-3,
        update_per_collect=100,
        collect_count=1000,
        action_size=6,
        # Users should add their own model path here. Model path should lead to a model.
        # Absolute path is recommended.
        # In DI-engine, it is ``exp_name/ckpt/ckpt_best.pth.tar``.
        expert_model_path='model_path_placeholder',
        # Path where to store the reward model
        reward_model_path='data_path_placeholder+/reward_model/ckpt/ckpt_best.pth.tar',
        # Users should add their own data path here. Data path should lead to a file to store data or load the stored data.
        # Absolute path is recommended.
        # In DI-engine, it is usually located in ``exp_name`` directory
        # e.g. 'exp_name/expert_data.pkl'
        data_path='data_path_placeholder',
    ),
    policy=dict(
        cuda=True,
        priority=False,
        model=dict(
            obs_shape=[4, 84, 84],
            action_shape=6,
            encoder_hidden_size_list=[128, 128, 512],
        ),
        nstep=1,
        discount_factor=0.99,
        learn=dict(
            update_per_collect=10,
            batch_size=32,
            learning_rate=0.0001,
            target_update_freq=500,
        ),
        collect=dict(n_sample=96, ),
        eval=dict(evaluator=dict(eval_freq=4000, )),
        other=dict(
            eps=dict(
                type='exp',
                start=1.,
                end=0.05,
                decay=250000,
            ),
            replay_buffer=dict(replay_buffer_size=100000, ),
        ),
    ),
)
pong_dqn_gail_config = EasyDict(pong_dqn_gail_config)
main_config = pong_dqn_gail_config
pong_dqn_gail_create_config = dict(
    env=dict(
        type='atari',
        import_names=['dizoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='dqn'),
)
pong_dqn_gail_create_config = EasyDict(pong_dqn_gail_create_config)
create_config = pong_dqn_gail_create_config

if __name__ == '__main__':
    # or you can enter `ding -m serial_gail -c pong_gail_dqn_config.py -s 0`
    # then input the config you used to generate your expert model in the path mentioned above
    # e.g. pong_dqn_config.py
    from ding.entry import serial_pipeline_gail
    from dizoo.atari.config.serial.pong import pong_dqn_config, pong_dqn_create_config
    expert_main_config = pong_dqn_config
    expert_create_config = pong_dqn_create_config
    serial_pipeline_gail(
        (main_config, create_config), (expert_main_config, expert_create_config),
        max_env_step=1000000,
        seed=0,
        collect_data=True
    )