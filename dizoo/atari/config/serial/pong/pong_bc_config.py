from copy import deepcopy
from ding.entry import serial_pipeline
from easydict import EasyDict

pong_dqn_config = dict(
    exp_name='pong_expert_model',
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
        ),
        learn=dict(
            multi_gpu=False,
            bp_update_sync=False,
            train_epoch=200,
            batch_size=32,
            learning_rate=0.0001,
            decay_epoch=30,
            decay_rate=0.1,
            warmup_lr=1e-4,
            warmup_epoch=3,
            weight_decay=1e-4,
            learner=dict(
                log_show_freq=10,
                hook=dict(
                    log_show_after_iter=int(1e9),  # use user-defined hook, disable it
                    save_ckpt_after_iter=1000,
                )
            )
        ),
        collect=dict(n_sample=96, demonstration_model_path = '/Users/nieyunpeng/Documents/open-sourced-algorithms/BC_DI-engine/dizoo/atari/config/serial/pong/default_experiment/ckpt/iteration_0.pth.tar', demonstration_offline_data_path = '/Users/nieyunpeng/Documents/open-sourced-algorithms/BC_DI-engine/dizoo/behaviour_cloning/entry/expert_data.pkl'),
        eval=dict(
            batch_size=32,
            evaluator=dict(
                eval_freq=4000,
                multi_gpu=False,
                stop_value=dict(
                    loss=0.5,
                    acc=75.0,
                )
            )
        ),
    ),
)
pong_dqn_config = EasyDict(pong_dqn_config)
main_config = pong_dqn_config
pong_dqn_create_config = dict(
    env=dict(
        type='atari',
        import_names=['dizoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='dqn'),
)
pong_dqn_create_config = EasyDict(pong_dqn_create_config)
create_config = pong_dqn_create_config

if __name__ == '__main__':
    serial_pipeline((main_config, create_config), seed=0)
