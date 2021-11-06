from easydict import EasyDict
from ding.entry import serial_pipeline_reward_model

minigrid_ppo_rnd_config = dict(
    exp_name='PitfallNoFrameskip-v4',  #exp_name='minigrid_empty8_ppo_rnd',
    #exp_name='MiniGrid-DoorKey-16x16-v0-curiosity-real_version',  #exp_name='minigrid_empty8_ppo_rnd',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        env_id='PitfallNoFrameskip-v4',  #env_id='MiniGrid-Empty-8x8-v0', MiniGrid-FourRooms-v0, MiniGrid-DoorKey-16x16-v0
        stop_value=20,
        frame_stack=4,
    ),
    reward_model=dict(
        intrinsic_reward_type='add',  # 'assign'
        learning_rate=0.001,
        obs_shape=[4, 84, 84],
        batch_size=128,
        update_per_collect=10,
    ),
    policy=dict(
        cuda=False,
        model=dict(
            obs_shape=[4, 84, 84],
            action_shape=6,
            encoder_hidden_size_list=[64, 64, 128],
            actor_head_hidden_size=128,
            critic_head_hidden_size=128,
        ),
        learn=dict(
            update_per_collect=24,
            batch_size=128,
            # (bool) Whether to normalize advantage. Default to False.
            adv_norm=False,
            learning_rate=0.0002,
            # (float) loss weight of the value network, the weight of policy network is set to 1
            value_weight=0.5,
            # (float) loss weight of the entropy regularization, the weight of policy network is set to 1
            entropy_weight=0.015,
            clip_ratio=0.1,
        ),
        collect=dict(
            # (int) collect n_sample data, train model n_iteration times
            n_sample=1024,
            # (float) the trade-off factor lambda to balance 1step td and mc
            gae_lambda=0.95,
            discount_factor=0.99,
        ),
        eval=dict(evaluator=dict(eval_freq=1000, )),
        other=dict(replay_buffer=dict(
            replay_buffer_size=100000,
            max_use=3,
        ), ),
    ),
)
minigrid_ppo_rnd_config = EasyDict(minigrid_ppo_rnd_config)
main_config = minigrid_ppo_rnd_config
minigrid_ppo_rnd_create_config = dict(
    env=dict(
        type='atari',
        import_names=['dizoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='ppo_offpolicy'),
    reward_model=dict(type='icm'),  #'rnd'
)
minigrid_ppo_rnd_create_config = EasyDict(minigrid_ppo_rnd_create_config)
create_config = minigrid_ppo_rnd_create_config

if __name__ == "__main__":
    serial_pipeline_reward_model([main_config, create_config], seed=0)
