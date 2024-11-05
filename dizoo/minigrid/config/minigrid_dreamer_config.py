from easydict import EasyDict

from ding.entry import serial_pipeline_dreamer

cuda = False
collector_env_num = 8
evaluator_env_num = 5
minigrid_dreamer_config = dict(
    exp_name='minigrid_dreamer_empty',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        # typical MiniGrid env id:
        # {'MiniGrid-Empty-8x8-v0', 'MiniGrid-FourRooms-v0', 'MiniGrid-DoorKey-8x8-v0','MiniGrid-DoorKey-16x16-v0'},
        # please refer to https://github.com/Farama-Foundation/MiniGrid for details.
        env_id='MiniGrid-Empty-8x8-v0',
        # env_id='MiniGrid-AKTDT-7x7-1-v0',
        max_step=100,
        stop_value=20,  # run fixed env_steps
        # stop_value=0.96,
        flat_obs=True,
        full_obs=True,
        onehot_obs=True,
        move_bonus=True,
    ),
    policy=dict(
        cuda=cuda,
        # it is better to put random_collect_size in policy.other
        random_collect_size=2500,
        model=dict(
            action_shape=7,
            # encoder_hidden_size_list=[256, 128, 64, 64],
            # critic_head_hidden_size=64,
            # actor_head_hidden_size=64,
            actor_dist='onehot',
        ),
        learn=dict(
            lambda_=0.95,
            learning_rate=3e-5,
            batch_size=16,
            batch_length=64,
            imag_sample=True,
            discount=0.997,
            reward_EMA=True,
        ),
        collect=dict(
            n_sample=1,
            unroll_len=1,
            action_size=7,  # has to be specified
            collect_dyn_sample=True,
        ),
        eval=dict(evaluator=dict(eval_freq=5000, )),
        other=dict(
            # environment buffer
            replay_buffer=dict(replay_buffer_size=500000, periodic_thruput_seconds=60),
        ),
    ),
    world_model=dict(
        pretrain=100,
        train_freq=2,
        cuda=cuda,
        model=dict(
            state_size=1344,
            obs_type='vector',
            action_size=7,
            action_type='discrete',
            encoder_hidden_size_list=[256, 128, 64, 64],
            reward_size=1,
            batch_size=16,
        ),
    ),
)

minigrid_dreamer_config = EasyDict(minigrid_dreamer_config)

minigrid_create_config = dict(
    env=dict(
        type='minigrid',
        import_names=['dizoo.minigrid.envs.minigrid_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(
        type='dreamer',
        import_names=['ding.policy.mbpolicy.dreamer'],
    ),
    replay_buffer=dict(type='sequence', ),
    world_model=dict(
        type='dreamer',
        import_names=['ding.world_model.dreamer'],
    ),
)
minigrid_create_config = EasyDict(minigrid_create_config)

if __name__ == '__main__':
    serial_pipeline_dreamer((minigrid_dreamer_config, minigrid_create_config), seed=0, max_env_step=500000)
