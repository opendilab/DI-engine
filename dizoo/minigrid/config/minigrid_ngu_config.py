import torch
from easydict import EasyDict

from ding.entry import serial_pipeline_reward_model_ngu

print(torch.cuda.is_available(), torch.__version__)
collector_env_num = 32  # TODO 32 5
evaluator_env_num = 5
nstep = 5
minigrid_ppo_rnd_config = dict(
    # exp_name='debug_minigrid_empty8_ngu_ul98_er01_n32_rbs5e4_fixepseval',
    # exp_name='debug_minigrid_fourrooms_ngu_ul98_er01_n32_rbs5e4_fixepseval_seed3',
    exp_name='debug_minigrid_doorkey_ngu_ul298_er01_n32_rbs3e4_fixepseval_ee0.1_ed0.1_upc16_seed0',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=5,
        # env_id='MiniGrid-Empty-8x8-v0',
        # env_id='MiniGrid-FourRooms-v0',
        env_id='MiniGrid-DoorKey-16x16-v0',
        stop_value=0.96,
    ),
    rnd_reward_model=dict(
        intrinsic_reward_type='add',  # 'assign'
        learning_rate=5e-4,
        obs_shape=2739,
        action_shape=7,
        batch_size=320,  # transitions
        update_per_collect=int(10),  # 32*100/320=10
        only_use_last_five_frames_for_icm_rnd=False,  # TODO(pu): True
        clear_buffer_per_iters=10,
        nstep=nstep,
        hidden_size_list=[128, 128, 64],
        type='rnd-ngu',

    ),
    episodic_reward_model=dict(
        intrinsic_reward_type='add',
        learning_rate=5e-4,
        obs_shape=2739,
        action_shape=7,
        batch_size=320,  # transitions
        update_per_collect=int(10),  # 32*100/64=50
        only_use_last_five_frames_for_icm_rnd=False,
        clear_buffer_per_iters=10,
        nstep=nstep,
        hidden_size_list=[128, 128, 64],
        type='episodic',
    ),
    policy=dict(
        continuous=False,
        on_policy=False,
        cuda=True,
        priority=True,
        priority_IS_weight=True,
        discount_factor=0.997,
        burnin_step=2,
        nstep=nstep,
        unroll_len=298,  # TODO(pu): according to the episode length
        model=dict(
            obs_shape=2739,
            action_shape=7,
            encoder_hidden_size_list=[128, 128, 512],
            collector_env_num=collector_env_num,
        ),
        learn=dict(
            # update_per_collect=8,
            update_per_collect=16,  # TODO(pu) 32*32/64=16
            batch_size=64,
            learning_rate=1e-4,
            target_update_theta=0.001,
        ),
        collect=dict(
            # NOTE it is important that don't include key n_sample here, to make sure self._traj_len=INF
            each_iter_n_sample=32,
            env_num=collector_env_num,
        ),
        eval=dict(env_num=evaluator_env_num, ),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.1,  # TODO(pu) 0.05
                decay=1e7,  # TODO(pu) 1e5
            ),
            replay_buffer=dict(replay_buffer_size=int(3e4),
                               # (Float type) How much prioritization is used: 0 means no prioritization while 1 means full prioritization
                               alpha=0.6,
                               # (Float type)  How much correction is used: 0 means no correction while 1 means full correction
                               beta=0.4,
                               )
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
    # env_manager=dict(type='base'),
    env_manager=dict(type='subprocess'),
    policy=dict(type='ngu'),
    rnd_reward_model=dict(type='rnd-ngu'),
    episodic_reward_model=dict(type='episodic'),
    collector=dict(type='sample_ngu',)
)
minigrid_ppo_rnd_create_config = EasyDict(minigrid_ppo_rnd_create_config)
create_config = minigrid_ppo_rnd_create_config

if __name__ == "__main__":
    serial_pipeline_reward_model_ngu([main_config, create_config], seed=0)
