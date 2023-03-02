from copy import deepcopy
from easydict import EasyDict

asterix_mdqn_config = dict(
    exp_name='asterix_mdqn_seed0',
    env=dict(
        collector_env_num=1,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=20000,
        env_id='Asterix-v0',
        #'ALE/SpaceInvaders-v5' is available. But special setting is needed after gym make.
        frame_stack=4,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        cuda=True,
        priority=False,
        model=dict(
            obs_shape=[4, 84, 84],
            action_shape=9,
            encoder_hidden_size_list=[32, 32, 64, 512],
        ),
        nstep=1,
        discount_factor=0.99,
        entropy_tau=0.03,
        m_alpha=0.9,
        learn=dict(
            update_per_collect=1,
            batch_size=32,
            learning_rate=0.00005,
            target_update_freq=2000,
            learner=dict(hook=dict(save_ckpt_after_iter=1000000, ))
        ),
        collect=dict(n_sample=4, ),
        eval=dict(evaluator=dict(eval_freq=4000, )),
        other=dict(
            eps=dict(
                type='exp',
                start=0.01,
                end=0.001,
                decay=250000,
            ),
            replay_buffer=dict(replay_buffer_size=400000, ),
        ),
    ),
)
asterix_mdqn_config = EasyDict(asterix_mdqn_config)
main_config = asterix_mdqn_config
asterix_mdqn_create_config = dict(
    env=dict(
        type='atari',
        import_names=['dizoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='mdqn'),
)
asterix_mdqn_create_config = EasyDict(asterix_mdqn_create_config)
create_config = asterix_mdqn_create_config

if __name__ == '__main__':
    # or you can enter ding -m serial -c spaceinvaders_dqn_config.py -s 0
    from ding.entry import serial_pipeline
    serial_pipeline((main_config, create_config), seed=0)
