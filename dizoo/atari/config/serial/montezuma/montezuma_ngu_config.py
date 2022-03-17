import torch
from easydict import EasyDict

from ding.entry import serial_pipeline_reward_model_ngu

print(torch.cuda.is_available(), torch.__version__)
collector_env_num = 32
evaluator_env_num = 5
nstep = 5
montezuma_ppo_rnd_config = dict(
    # Note: 
    # 1. at least 1e9 timesteps, i.e., 1000 million, the reward may increase, please be patient.
    # 2. the larger unroll_lenth and replay buffer size may have better results, but also require more memory.
    # exp_name='debug_montezuma_ngu_ul298_er01_n32_rlbs1e4',
    # exp_name='debug_montezuma_ngu_ul40_er01_n32_rlbs1e4',
    # exp_name='debug_montezuma_ngu_ul98_er01_n32_rlbs3e3_maxstep1e6',
    exp_name='debug_montezuma_ngu_ul198_er01_n32_rlbs2e3_maxstep1e6',


    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=5,
        env_id='MontezumaRevengeNoFrameskip-v4',
        stop_value=int(1e5),
        frame_stack=4,
    ),
    rnd_reward_model=dict(
        intrinsic_reward_type='add',  # 'assign'
        learning_rate=0.001,
        obs_shape=[4, 84, 84],
        action_shape=18,
        batch_size=320,
        update_per_collect=int(10),  # 32*100/64=50
        only_use_last_five_frames_for_icm_rnd=False,  # TODO(pu): True
        clear_buffer_per_iters=10,
        nstep=nstep,
        hidden_size_list=[128, 128, 64],
        type='rnd-ngu',
    ),
    episodic_reward_model=dict(
        intrinsic_reward_type='add',
        learning_rate=0.001,
        obs_shape=[4, 84, 84],
        action_shape=18,
        batch_size=320,
        update_per_collect=int(10),  # 32*100/64=50
        only_use_last_five_frames_for_icm_rnd=False,  # TODO(pu): True
        clear_buffer_per_iters=10,
        nstep=nstep,
        hidden_size_list=[128, 128, 64],
        type='episodic',
    ),
    policy=dict(
        cuda=True,
        on_policy=False,
        priority=True,
        priority_IS_weight=True,
        discount_factor=0.997,
        burnin_step=2,
        nstep=nstep,
        unroll_len=198,#40,#298,
        model=dict(
            obs_shape=[4, 84, 84],
            action_shape=18,
            encoder_hidden_size_list=[128, 128, 512],
            collector_env_num=collector_env_num,
        ),
        learn=dict(
            update_per_collect=8,
            batch_size=64,
            learning_rate=0.0005,
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
                end=0.05,
                decay=1e5,
            ),
            replay_buffer=dict(
                replay_buffer_size=int(2e3),
                # (Float type) How much prioritization is used: 0 means no prioritization while 1 means full prioritization
                alpha=0.6,
                # (Float type)  How much correction is used: 0 means no correction while 1 means full correction
                beta=0.4,
            )
        ),
    ),
)
montezuma_ppo_rnd_config = EasyDict(montezuma_ppo_rnd_config)
main_config = montezuma_ppo_rnd_config
montezuma_ppo_rnd_create_config = dict(
    env=dict(
        type='atari',
        import_names=['dizoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='base'),
    # env_manager=dict(type='subprocess'),
    policy=dict(type='ngu'),
    # reward_model=dict(type='rnd'),
    rnd_reward_model=dict(type='rnd-ngu'),
    episodic_reward_model=dict(type='episodic'),
    collector=dict(type='sample_ngu', )
)
montezuma_ppo_rnd_create_config = EasyDict(montezuma_ppo_rnd_create_config)
create_config = montezuma_ppo_rnd_create_config

if __name__ == "__main__":
    serial_pipeline_reward_model_ngu([main_config, create_config], seed=0)
