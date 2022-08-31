from easydict import EasyDict

pong_dqn_stdim_config = dict(
    exp_name='pong_dqn_stdim_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=20,
        env_id='Pong-v4',
        #'ALE/Pong-v5' is available. But special setting is needed after gym make.
        frame_stack=4,
    ),
    policy=dict(
        cuda=True,
        priority=False,
        model=dict(
            obs_shape=[4, 84, 84],
            action_shape=6,
            encoder_hidden_size_list=[128, 128, 512],
        ),
        aux_model=dict(
            encode_shape=64,
            heads=[1, 1],
            loss_type='infonce',
            temperature=1.0,
        ),
        # the weight of the auxiliary loss to the TD loss
        aux_loss_weight=0.003,
        nstep=3,
        discount_factor=0.99,
        learn=dict(
            update_per_collect=10,
            batch_size=128,
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
pong_dqn_stdim_config = EasyDict(pong_dqn_stdim_config)
main_config = pong_dqn_stdim_config
pong_dqn_stdim_create_config = dict(
    env=dict(
        type='atari',
        import_names=['dizoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='dqn_stdim'),
)
pong_dqn_stdim_create_config = EasyDict(pong_dqn_stdim_create_config)
create_config = pong_dqn_stdim_create_config

if __name__ == '__main__':
    # or you can enter `ding -m serial -c pong_dqn_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline((main_config, create_config), seed=0)
