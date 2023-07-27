from easydict import EasyDict

cartpole_sil_config = dict(
    exp_name='cartpole_sil_a2c_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=195,
    ),
    policy=dict(
        cuda=False,
        sil_update_per_collect=5,
        model=dict(
            obs_shape=4,
            action_shape=2,
            encoder_hidden_size_list=[128, 128, 64],
        ),
        learn=dict(
            batch_size=40,
            learning_rate=0.001,
        ),
        collect=dict(
            # (int) collect n_sample data, train model n_iteration times
            n_sample=80,
            # (float) the trade-off factor lambda to balance 1step td and mc
            gae_lambda=0.95,
        ),
        eval=dict(evaluator=dict(eval_freq=50, )),
    ),
)
cartpole_sil_config = EasyDict(cartpole_sil_config)
main_config = cartpole_sil_config

cartpole_sil_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='sil_a2c'),
)
cartpole_sil_create_config = EasyDict(cartpole_sil_create_config)
create_config = cartpole_sil_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial_onpolicy -c cartpole_sil_config.py -s 0`
    from ding.entry import serial_pipeline_sil
    serial_pipeline_sil((main_config, create_config), seed=0)
