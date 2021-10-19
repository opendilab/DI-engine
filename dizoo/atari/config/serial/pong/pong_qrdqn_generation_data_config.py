from copy import deepcopy
from ding.entry import serial_pipeline
from easydict import EasyDict

pong_qrdqn_config = dict(
    env=dict(
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=20,
        env_id='PongNoFrameskip-v4',
        frame_stack=4,
        manager=dict(shared_memory=False, )
    ),
    policy=dict(
        cuda=True,
        priority=False,
        model=dict(
            obs_shape=[4, 84, 84],
            action_shape=6,
            encoder_hidden_size_list=[128, 128, 512],
            num_quantiles=64,
        ),
        nstep=1,
        discount_factor=0.99,
        learn=dict(
            update_per_collect=10,
            batch_size=32,
            learning_rate=0.0001,
            target_update_freq=500,
            learner=dict(
                load_path='./expert/ckpt/ckpt_best.pth.tar',
                hook=dict(
                    load_ckpt_before_run='./expert/ckpt/ckpt_best.pth.tar',
                    save_ckpt_after_run=False,
                )
            ),
        ),
        collect=dict(
            n_sample=100,
            data_type='hdf5',
            save_path='./expert/expert.pkl',
        ),
        eval=dict(evaluator=dict(eval_freq=4000, )),
        other=dict(
            eps=dict(
                type='exp',
                start=1.,
                end=0.05,
                decay=250000,
                collect=0.2,
            ),
            replay_buffer=dict(replay_buffer_size=100000, ),
        ),
    ),
)
pong_qrdqn_config = EasyDict(pong_qrdqn_config)
main_config = pong_qrdqn_config
pong_qrdqn_create_config = dict(
    env=dict(
        type='atari',
        import_names=['dizoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='qrdqn'),
)
pong_qrdqn_create_config = EasyDict(pong_qrdqn_create_config)
create_config = pong_qrdqn_create_config

if __name__ == '__main__':
    serial_pipeline((main_config, create_config), seed=0)
