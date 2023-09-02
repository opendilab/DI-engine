from easydict import EasyDict

from ding.entry import serial_pipeline_dreamer

cuda = False

cheetah_run_dreamer_config = dict(
    exp_name='dmc2gym_cheetah_run_dreamer',
    env=dict(
        env_id='dmc2gym_cheetah_run',
        domain_name='cheetah',
        task_name='run',
        frame_skip=1,
        warp_frame=True,
        scale=True,
        clip_rewards=False,
        action_repeat=2,
        frame_stack=1,
        from_pixels=True,
        resize=64,
        collector_env_num=1,
        evaluator_env_num=1,
        n_evaluator_episode=1,
        stop_value=1000,  # 1000
    ),
    policy=dict(
        cuda=cuda,
        # it is better to put random_collect_size in policy.other
        random_collect_size=2500,
        model=dict(
            obs_shape=(3, 64, 64),
            action_shape=6,
            actor_dist='normal',
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
            action_size=6,  # has to be specified
            collect_dyn_sample=True,
        ),
        command=dict(),
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
            state_size=(3, 64, 64),  # has to be specified
            action_size=6,  # has to be specified
            reward_size=1,
            batch_size=16,
        ),
    ),
)

cheetah_run_dreamer_config = EasyDict(cheetah_run_dreamer_config)

cheetah_run_create_config = dict(
    env=dict(
        type='dmc2gym',
        import_names=['dizoo.dmc2gym.envs.dmc2gym_env'],
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
cheetah_run_create_config = EasyDict(cheetah_run_create_config)

if __name__ == '__main__':
    serial_pipeline_dreamer((cheetah_run_dreamer_config, cheetah_run_create_config), seed=0, max_env_step=500000)
