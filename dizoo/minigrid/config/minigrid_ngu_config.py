from easydict import EasyDict
from ding.entry import serial_pipeline_reward_model
import torch
print(torch.cuda.is_available(),torch.__version__)
collector_env_num = 8
evaluator_env_num = 5
nstep=5
minigrid_ppo_rnd_config = dict(   
    exp_name='minigrid_empty8_r2d2_rnd_noadd',
    # exp_name='minigrid_empty8_r2d2_debug',
    env=dict(
        collector_env_num=collector_env_num ,
        evaluator_env_num=evaluator_env_num ,
        n_evaluator_episode=5,
        env_id='MiniGrid-Empty-8x8-v0',
        stop_value=0.96,
    ),
    reward_model=dict(
        intrinsic_reward_type='add', #add',  # 'assign'
        learning_rate=0.001,
        obs_shape=2739,
        batch_size=32,
        update_per_collect=10,
        nstep=nstep,
    ),
    policy=dict(
        continuous=False,
        on_policy=False, 
        cuda=True,
        priority=False,
        discount_factor=0.997,
        burnin_step=20,
        nstep=nstep,
        unroll_len=80, #
        model=dict(
            obs_shape=2739,
            action_shape=7,
            encoder_hidden_size_list=[256, 128, 64, 64],
        ),
        learn=dict(
            update_per_collect=20,#4,
            batch_size=32, #64,
            learning_rate=0.0005,
            value_weight=0.5,
            entropy_weight=0.001,
            clip_ratio=0.2,
            adv_norm=False,
            target_update_freq=500,#100,
        ),
        collect=dict(
            # n_sample=128,
            n_sample=32,
            # unroll_len=1,
            # unroll_len=40,
            env_num=collector_env_num,
        ),
        eval=dict(env_num=evaluator_env_num, ),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.05,
                decay=10000,
            ), replay_buffer=dict(replay_buffer_size=1000000, ) #10000
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
    policy=dict(type='ngu'),
    reward_model=dict(type='rnd'),
)
minigrid_ppo_rnd_create_config = EasyDict(minigrid_ppo_rnd_create_config)
create_config = minigrid_ppo_rnd_create_config

if __name__ == "__main__":
    serial_pipeline_reward_model([main_config, create_config], seed=0)
