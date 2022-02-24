import torch
from easydict import EasyDict

from ding.entry import serial_pipeline_reward_model_ngu

print(torch.cuda.is_available(), torch.__version__)
collector_env_num = 32
evaluator_env_num = 5
nstep = 5
pong_ppo_rnd_config = dict(
    exp_name='debug_pong_ngu_ul98_er01_rlbs2e4_n32',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=5,
        env_id='PongNoFrameskip-v4',
        stop_value=20,
        frame_stack=4,
    ),
    rnd_reward_model=dict(
        intrinsic_reward_type='add',  # 'assign'
        learning_rate=1e-4,
        obs_shape=[4, 84, 84],
        action_shape=6,
        batch_size=64,
        update_per_collect=int(50 / 2),  # 32*100/64=50
        only_use_last_five_frames_for_icm_rnd=False,
        clear_buffer_per_iters=10,
        # update_per_collect=3,  # 32*5/64=3
        # only_use_last_five_frames_for_icm_rnd=True,
        nstep=nstep,
        hidden_size_list=[128, 128, 64],
        type='rnd',
    ),
    episodic_reward_model=dict(
        intrinsic_reward_type='add',
        learning_rate=1e-4,
        obs_shape=[4, 84, 84],
        action_shape=6,
        batch_size=64,
        update_per_collect=int(50 / 2),  # 32*100/64=50
        only_use_last_five_frames_for_icm_rnd=False,
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
        burnin_step=10,
        nstep=nstep,
        unroll_len=98,  # TODO(pu): 40
        model=dict(
            obs_shape=[4, 84, 84],
            action_shape=6,
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
                replay_buffer_size=20000,  # TODO(pu)
                # (Float type) How much prioritization is used: 0 means no prioritization while 1 means full prioritization
                alpha=0.6,
                # (Float type)  How much correction is used: 0 means no correction while 1 means full correction
                beta=0.4,
            )
        ),
    ),
)
pong_ppo_rnd_config = EasyDict(pong_ppo_rnd_config)
main_config = pong_ppo_rnd_config
pong_ppo_rnd_create_config = dict(
    env=dict(
        type='atari',
        import_names=['dizoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='base'),
    # env_manager=dict(type='subprocess'),
    policy=dict(type='ngu'),
    # reward_model=dict(type='rnd'),
    rnd_reward_model=dict(type='rnd'),
    episodic_reward_model=dict(type='episodic'),
    collector=dict(type='sample_ngu', )
)
pong_ppo_rnd_create_config = EasyDict(pong_ppo_rnd_create_config)
create_config = pong_ppo_rnd_create_config

if __name__ == "__main__":
    serial_pipeline_reward_model_ngu([main_config, create_config], seed=0)
