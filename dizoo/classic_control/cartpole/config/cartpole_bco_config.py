from easydict import EasyDict

cartpole_expert_model_config = dict(
    exp_name='cartpole_expert_model',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=195,
        replay_path='cartpole_dqn/video',
    ),
    policy=dict(
        cuda=True,
        model=dict(
            obs_shape=4,
            action_shape=2,
            encoder_hidden_size_list=[128, 128, 64],
            dueling=True,
        ),
        learn=dict(
            multi_gpu=False,
            bp_update_sync=False,
            train_epoch=200,
            batch_size=32,
            learning_rate=0.01,
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
        collect=dict(
            n_sample=96,
            demonstration_model_path='/home/lisong/Projects/BCO/DI-engine/cartpole_dqn/ckpt/ckpt_best.pth.tar',
            demonstration_offline_data_path='/home/lisong/Projects/BCO/DI-engine/cartpole_dqn/expert_data.pkl',
        ),
        eval=dict(batch_size=32, evaluator=dict(eval_freq=1, multi_gpu=False, stop_value=dict(
            loss=0.5,
            acc=75.0,
        ))),
    ),
)
cartpole_expert_model_config = EasyDict(cartpole_expert_model_config)
main_config = cartpole_expert_model_config
cartpole_expert_model_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='bc'),
)
cartpole_expert_model_create_config = EasyDict(cartpole_expert_model_create_config)
create_config = cartpole_expert_model_create_config

if __name__ == "__main__":
    from ding.entry import serial_pipeline_bco
    serial_pipeline_bco(main_config, create_config, 0)
