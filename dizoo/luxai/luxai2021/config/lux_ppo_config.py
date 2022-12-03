from easydict import EasyDict
from dizoo.luxai.luxai2021.game.constants import Constants, GAME_CONSTANTS


# DI-Engine uses EasyDict for configuration, by convention
lux_ppo_config = EasyDict(dict(
    exp_name='lux_ppo_config',

    # Lux2021 Match Config
    mapType= Constants.MAP_TYPES.RANDOM,
    storeReplay=True,
    seed=None,
    debug=False,
    debugDelay=500,
    runProfiler=False,
    compressReplay=False,
    debugAnnotations=False,
    statefulReplay=False,
    parameters=GAME_CONSTANTS["PARAMETERS"],

    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=195,        
    ),

    policy=dict(
        cuda=False,
        priority=True,
        discount_factor=0.97,
        nstep=3,
        model=dict(
            obs_shape=2,
            action_shape=3,
            encoder_hidden_size_list=[128, 128, 64],
        ),
        learn=dict(
            update_per_collect=3,
            batch_size=64,
            learning_rate=0.001,
            target_update_freq=100,
        ),
        collect=dict(
            n_sample=80,
            unroll_len=1,
        ),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.1,
                decay=10000,
            ), replay_buffer=dict(replay_buffer_size=20000, )
        ),
    ),
))

main_config = lux_ppo_config

lux_ppo_create_config = EasyDict(dict(
    env=dict(
        type='lux2021',
        import_names=['dizoo.luxai.luxai2021.env.lux_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='ppo'),
))

create_config = lux_ppo_create_config

if __name__ == "__main__":
    from ding.entry import serial_pipeline
    serial_pipeline((main_config, create_config), seed=0)