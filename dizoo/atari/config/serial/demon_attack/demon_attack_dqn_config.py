from easydict import EasyDict

demon_attack_dqn_config = dict(
    exp_name='DemonAttack_dqn_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=1e6,
        env_id='DemonAttackNoFrameskip-v4',
        frame_stack=4,
    ),
    policy=dict(
        cuda=True,
        priority=False,
        model=dict(
            obs_shape=[4, 84, 84],
            action_shape=6,
            encoder_hidden_size_list=[128, 128, 512],
            noise=True,
        ),
        nstep=3,
        discount_factor=0.99,
        learn=dict(
            update_per_collect=10,
            batch_size=32,
            learning_rate=0.0001,
            target_update_freq=500,
        ),
        noisy_net=True,
        collect=dict(n_sample=96),
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
demon_attack_dqn_config = EasyDict(demon_attack_dqn_config)
main_config = demon_attack_dqn_config
demon_attack_dqn_create_config = dict(
    env=dict(
        type='atari',
        import_names=['dizoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='dqn'),
)
demon_attack_dqn_create_config = EasyDict(demon_attack_dqn_create_config)
create_config = demon_attack_dqn_create_config

if __name__ == '__main__':
    # or you can enter `ding -m serial -c demon_attack_dqn_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline((main_config, create_config), seed=0, max_env_step=int(10e6))
