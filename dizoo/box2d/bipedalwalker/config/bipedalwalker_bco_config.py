from easydict import EasyDict

bipedalwalker_bco_config = dict(
    exp_name='bipedalwalker_bco_seed0',
    env=dict(
        env_id='BipedalWalker-v3',
        collector_env_num=8,
        evaluator_env_num=5,
        # (bool) Scale output action into legal range.
        act_scale=True,
        n_evaluator_episode=5,
        stop_value=300,
        rew_clip=True,
        # The path to save the game replay
        replay_path=None,
    ),
    policy=dict(
        # Whether to use cuda for network.
        cuda=True,
        continuous=True,
        loss_type='l1_loss',
        model=dict(
            obs_shape=24,
            action_shape=4,
            action_space='regression',
            actor_head_hidden_size=128,
        ),
        learn=dict(
            train_epoch=30,
            batch_size=128,
            learning_rate=0.01,
            weight_decay=1e-4,
            decay_epoch=1000,
            decay_rate=0.5,
            warmup_lr=1e-4,
            warmup_epoch=3,
            optimizer='SGD',
            lr_decay=True,
            momentum=0.9,
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
        eval=dict(evaluator=dict(eval_freq=100, )),
        other=dict(replay_buffer=dict(replay_buffer_size=100000, ), ),
    ),
    bco=dict(
        learn=dict(idm_batch_size=128, idm_learning_rate=0.001, idm_weight_decay=0, idm_train_epoch=50),
        model=dict(
            action_space='regression',
            idm_encoder_hidden_size_list=[60, 80, 100, 40],
        ),
        alpha=0.2,
    )
)

bipedalwalker_bco_config = EasyDict(bipedalwalker_bco_config)
main_config = bipedalwalker_bco_config

bipedalwalker_bco_create_config = dict(
    env=dict(
        type='bipedalwalker',
        import_names=['dizoo.box2d.bipedalwalker.envs.bipedalwalker_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='bc'),
    collector=dict(type='episode'),
)
bipedalwalker_bco_create_config = EasyDict(bipedalwalker_bco_create_config)
create_config = bipedalwalker_bco_create_config

if __name__ == "__main__":
    from ding.entry import serial_pipeline_bco
    from dizoo.box2d.bipedalwalker.config import bipedalwalker_sac_config, bipedalwalker_sac_create_config
    expert_main_config = bipedalwalker_sac_config
    expert_create_config = bipedalwalker_sac_create_config
    serial_pipeline_bco(
        [main_config, create_config], [expert_main_config, expert_create_config], seed=0, max_env_step=2000000
    )
