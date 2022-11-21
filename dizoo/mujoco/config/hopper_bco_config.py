from easydict import EasyDict

hopper_bco_config = dict(
    exp_name='hopper_bco_seed0',
    env=dict(
        env_id='Hopper-v3',
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        collector_env_num=1,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=6000,
    ),
    policy=dict(
        # Whether to use cuda for network.
        cuda=True,
        continuous=True,
        loss_type='l1_loss',
        model=dict(
            obs_shape=11,
            action_shape=3,
            action_space='regression',
            actor_head_hidden_size=128,
        ),
        learn=dict(
            train_epoch=20,
            batch_size=128,
            learning_rate=0.001,
            weight_decay=1e-4,
            momentum=0.9,
            decay_epoch=30,
            decay_rate=1,
            warmup_lr=1e-4,
            warmup_epoch=3,
            optimizer='SGD',
            lr_decay=True,
        ),
        collect=dict(
            n_episode=100,
            # control the number (alpha*n_episode) of post-demonstration environment interactions at each iteration.
            # Notice: alpha * n_episode > collector_env_num
            model_path='abs model path',  # expert model path
            data_path='abs data path',  # expert data path
            noise=True,
            noise_sigma=dict(
                start=0.5,
                end=0.1,
                decay=1000000,
                type='exp',
            ),
            noise_range=dict(
                min=-1,
                max=1,
            ),
        ),
        eval=dict(evaluator=dict(eval_freq=40, )),
        other=dict(replay_buffer=dict(replay_buffer_size=100000, ), ),
    ),
    bco=dict(
        learn=dict(idm_batch_size=256, idm_learning_rate=0.001, idm_weight_decay=0, idm_train_epoch=20),
        model=dict(
            action_space='regression',
            idm_encoder_hidden_size_list=[60, 80, 100, 40],
        ),
        alpha=0.2,
    )
)

hopper_bco_config = EasyDict(hopper_bco_config)
main_config = hopper_bco_config

hopper_bco_create_config = dict(
    env=dict(
        type='mujoco',
        import_names=['dizoo.mujoco.envs.mujoco_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='bc'),
    collector=dict(type='episode'),
)
hopper_bco_create_config = EasyDict(hopper_bco_create_config)
create_config = hopper_bco_create_config

if __name__ == "__main__":
    from ding.entry import serial_pipeline_bco
    from dizoo.mujoco.config.hopper_sac_config import hopper_sac_config, hopper_sac_create_config
    expert_main_config = hopper_sac_config
    expert_create_config = hopper_sac_create_config
    serial_pipeline_bco(
        [main_config, create_config], [expert_main_config, expert_create_config], seed=0, max_env_step=3000000
    )
