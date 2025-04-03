from easydict import EasyDict

pong_dqn_envpool_config = dict(
    exp_name='pong_dqn_envpool_seed0',
    env=dict(
        collector_env_num=8,
        collector_batch_size=8,
        evaluator_env_num=8,
        evaluator_batch_size=8,
        n_evaluator_episode=8,
        stop_value=21,
        env_id='Pong-v5',
        #'ALE/Pong-v5' is available. But special setting is needed after gym make.
        frame_stack=4,
    ),
    nstep = 3,
    policy=dict(
        cuda=True,
        priority=False,
        random_collect_size=50000,
        model=dict(
            obs_shape=[4, 84, 84],
            action_shape=6,
            encoder_hidden_size_list=[128, 128, 512],
        ),
        nstep=3,
        discount_factor=0.99,
        learn=dict(
            update_per_collect=2,
            batch_size=32,
            learning_rate=0.0001,
            # If updating target network by replacement, \
            # target_update_freq should be larger than 0. \
            # If updating target network by changing several percentage of the origin weights, \
            # target_update_freq should be 0 and target_theta should be set.
            target_update_freq=None,
            target_theta=0.04,
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
pong_dqn_envpool_config = EasyDict(pong_dqn_envpool_config)
main_config = pong_dqn_envpool_config
pong_dqn_envpool_create_config = dict(
    env=dict(
        type='atari',
        import_names=['dizoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='envpool'),
    policy=dict(type='dqn'),
    replay_buffer=dict(type='deque'),
)
pong_dqn_envpool_create_config = EasyDict(pong_dqn_envpool_create_config)
create_config = pong_dqn_envpool_create_config

if __name__ == '__main__':
    # or you can enter `ding -m serial -c pong_dqn_envpool_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline((main_config, create_config), seed=0)
