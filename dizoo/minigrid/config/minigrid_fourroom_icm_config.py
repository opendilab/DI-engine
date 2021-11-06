from easydict import EasyDict
from ding.entry import serial_pipeline_reward_model

minigrid_ppo_rnd_config = dict(
    exp_name='MiniGrid-FourRooms-v0',  #exp_name='minigrid_empty8_ppo_rnd',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        env_id='MiniGrid-FourRooms-v0',  #env_id='MiniGrid-Empty-8x8-v0', MiniGrid-FourRooms-v0, MiniGrid-DoorKey-16x16-v0
        stop_value=0.96,
        #render = True,
        replay_path = '/home/SENSETIME/zhoutong/di_env/DI-engine/MiniGrid-FourRooms-v0/vedio'
    ),
    reward_model=dict(
        intrinsic_reward_type='add',  # 'assign'
        learning_rate=0.001,
        obs_shape=2739,
        batch_size=32,
        update_per_collect=10,
    ),
    policy=dict(
        cuda=False,
        model=dict(
            obs_shape=2739,
            action_shape=7,
            encoder_hidden_size_list=[256, 128, 64, 64],
        ),
        learn=dict(
            update_per_collect=4,
            batch_size=64,
            learning_rate=0.0003,
            value_weight=0.5,
            entropy_weight=0.001,
            clip_ratio=0.2,
            adv_norm=False,
        ),
        collect=dict(
            n_sample=128,
            unroll_len=1,
            discount_factor=0.99,
            gae_lambda=0.95,
        ),
    ),
)
minigrid_ppo_rnd_config = EasyDict(minigrid_ppo_rnd_config)
main_config = minigrid_ppo_rnd_config
minigrid_ppo_rnd_create_config = dict(
    env=dict(
        type='minigrid',
        import_names=['dizoo.minigrid.envs.minigrid_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='ppo_offpolicy'),
    reward_model=dict(type='icm'),  #'rnd'
)
minigrid_ppo_rnd_create_config = EasyDict(minigrid_ppo_rnd_create_config)
create_config = minigrid_ppo_rnd_create_config

if __name__ == "__main__":
    serial_pipeline_reward_model([main_config, create_config], seed=0)
