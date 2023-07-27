from easydict import EasyDict

collector_env_num = 4
evaluator_env_num = 4
minigrid_sil_config = dict(
    exp_name='minigrid_sil_a2c_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        # typical MiniGrid env id:
        # {'MiniGrid-Empty-8x8-v0', 'MiniGrid-FourRooms-v0', 'MiniGrid-DoorKey-8x8-v0','MiniGrid-DoorKey-16x16-v0'},
        # please refer to https://github.com/Farama-Foundation/MiniGrid for details.
        env_id='MiniGrid-DoorKey-8x8-v0',
        n_evaluator_episode=5,
        max_step=300,
        stop_value=0.96,
    ),
    policy=dict(
        cuda=False,
        sil_update_per_collect=1,
        model=dict(
            obs_shape=2835,
            action_shape=7,
            encoder_hidden_size_list=[256, 128, 64, 64],
        ),
        learn=dict(
            batch_size=64,
            learning_rate=0.0003,
            value_weight=0.5,
            entropy_weight=0.001,
            adv_norm=True,
        ),
        collect=dict(
            n_sample=128,
            discount_factor=0.99,
            gae_lambda=0.95,
        ),
    ),
)
minigrid_sil_config = EasyDict(minigrid_sil_config)
main_config = minigrid_sil_config

minigrid_sil_create_config = dict(
    env=dict(
        type='minigrid',
        import_names=['dizoo.minigrid.envs.minigrid_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='sil_a2c'),
)
minigrid_sil_create_config = EasyDict(minigrid_sil_create_config)
create_config = minigrid_sil_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial_onpolicy -c minigrid_sil_a2c_config.py -s 0`
    from ding.entry import serial_pipeline_sil
    serial_pipeline_sil((main_config, create_config), seed=0)
