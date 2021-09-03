from easydict import EasyDict
from ding.entry import serial_pipeline_reward_model_ngu
import torch

print(torch.cuda.is_available(), torch.__version__)
collector_env_num = 8
evaluator_env_num = 5
nstep = 5 #5
lunarlander_ngu_config = dict(
    exp_name='lunarlander_ngu_bs2_n5_ul40_upc4_tuf100_ed1e4_rbs5e4_debug',
    env=dict(
        collector_env_num=collector_env_num, 
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=5,
        stop_value=195,
    ),
    rnd_reward_model=dict(
        intrinsic_reward_type='add',  #add',  # 'assign'
        learning_rate=0.001,
        obs_shape=8,
        action_shape = 4,
        batch_size=32,
        update_per_collect=10,
        nstep=nstep,
        hidden_size_list = [128, 128, 64],
        type='rnd',
    ),
    episodic_reward_model=dict(
        intrinsic_reward_type='add',  #add',  # 'assign'
        learning_rate=0.001,
        obs_shape=8,
        action_shape = 4,
        batch_size=32,
        update_per_collect=10,
        nstep=nstep,
        hidden_size_list = [128, 128, 64],
        type='episodic',
    ),
    policy=dict(
        continuous=False,
        on_policy=False,
        cuda=True,
        priority=False,
        discount_factor=0.99,#0.999,
        burnin_step=2, #TODO
        nstep=nstep,
        unroll_len=40,  # 80
        model=dict(
            obs_shape=8,
            action_shape=4,
            encoder_hidden_size_list=[128, 128, 64],
            collector_env_num=collector_env_num,#TODO
        ),
        learn=dict(
            update_per_collect=4,#20,  #4,
            batch_size=64,#32, 
            learning_rate=0.0005,
            value_weight=0.5,
            entropy_weight=0.001,
            clip_ratio=0.2,
            adv_norm=False,
            target_update_freq=100,  #100,
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
            ),
            replay_buffer=dict(replay_buffer_size=50000, )  #10000
        ),
    ),
)
lunarlander_ngu_config = EasyDict(lunarlander_ngu_config)
main_config = lunarlander_ngu_config
lunarlander_ngu_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['dizoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='base'),
    # env_manager=dict(type='subprocess'),
    policy=dict(type='ngu'),
    rnd_reward_model=dict(type='rnd'),
    episodic_reward_model=dict(type='episodic'),
    collector=dict(
        type='sample_ngu', # TODO
    )
)
lunarlander_ngu_create_config = EasyDict(lunarlander_ngu_create_config)
create_config = lunarlander_ngu_create_config

if __name__ == "__main__":
    serial_pipeline_reward_model_ngu([main_config, create_config], seed=0)
