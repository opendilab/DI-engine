from easydict import EasyDict

nstep = 3
lunarlander_bco_config = dict(
    exp_name='lunarlander_bco_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=8,
        env_id='LunarLander-v2',
        n_evaluator_episode=8,
        stop_value=200,
    ),
    policy=dict(
        # Whether to use cuda for network.
        cuda=True,
        continuous=False,
        loss_type='l1_loss',
        model=dict(
            obs_shape=8,
            action_shape=4,
            encoder_hidden_size_list=[512, 64],
            # Whether to use dueling head.
            dueling=True,
        ),
        # learn_mode config
        learn=dict(
            update_per_collect=10,
            train_epoch=20,
            batch_size=64,
            learning_rate=0.001,
            weight_decay=1e-4,
            decay_epoch=1000,
            decay_rate=0.5,
            warmup_lr=1e-4,
            warmup_epoch=3,
            optimizer='SGD',
            lr_decay=True,
            momentum=0.9,
        ),
        # collect_mode config
        collect=dict(
            n_episode=100,
            model_path='abs model path',  # expert model path
            data_path='abs data path',  # expert data path
        ),
        # eval_mode config
        eval=dict(evaluator=dict(eval_freq=50, )),
        # command_mode config
        other=dict(
            # Epsilon greedy with decay.
            eps=dict(
                # Decay type. Support ['exp', 'linear'].
                type='exp',
                start=0.95,
                end=0.1,
                decay=50000,
            ),
            replay_buffer=dict(replay_buffer_size=100000, )
        ),
    ),
    bco=dict(
        learn=dict(idm_batch_size=256, idm_learning_rate=0.001, idm_weight_decay=1e-4, idm_train_epoch=10),
        model=dict(idm_encoder_hidden_size_list=[60, 80, 100, 40], action_space='discrete'),
        alpha=0.2,
    )
)
lunarlander_bco_config = EasyDict(lunarlander_bco_config)
main_config = lunarlander_bco_config

lunarlander_bco_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['dizoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='bc'),
    collector=dict(type='episode'),
)
lunarlander_bco_create_config = EasyDict(lunarlander_bco_create_config)
create_config = lunarlander_bco_create_config

if __name__ == "__main__":
    from ding.entry import serial_pipeline_bco
    from dizoo.box2d.lunarlander.config import lunarlander_dqn_config, lunarlander_dqn_create_config
    expert_main_config = lunarlander_dqn_config
    expert_create_config = lunarlander_dqn_create_config
    serial_pipeline_bco(
        [main_config, create_config], [expert_main_config, expert_create_config], seed=0, max_env_step=2000000
    )
