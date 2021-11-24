from easydict import EasyDict

pong_dqn_gail_config = dict(
    exp_name='pong_dqn_gail',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=20,
        env_id='PongNoFrameskip-v4',
        frame_stack=4,
        manager=dict(shared_memory=False, )
    ),
    reward_model=dict(
        type='gail',
        input_size=[4, 84, 84],
        hidden_size=128,
        batch_size=64,
        learning_rate=1e-3,
        update_per_collect=100,
        expert_data_path='pong_dqn/expert_data_train.pkl',
        load_path='pong_dqn_gail/reward_model/ckpt/ckpt_last.pth.tar',  # state_dict of the reward model
        collect_count=1000,
        action_size=6
    ),
    policy=dict(
        load_path='pong_dqn_gail/ckpt/ckpt_best.pth.tar',
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
