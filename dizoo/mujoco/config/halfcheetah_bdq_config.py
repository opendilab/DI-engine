from easydict import EasyDict

halfcheetah_bdq_config = dict(
    exp_name='halfcheetah_bdq_seed0',
    env=dict(
        env_id='HalfCheetah-v3',
        norm_reward=dict(use_norm=False, ),
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=12000,
        action_bins_per_branch=2,
    ),
    policy=dict(
        cuda=False,
        priority=False,
        discount_factor=0.99,
        nstep=1,
        model=dict(
            obs_shape=17,
            num_branches=6,
            action_bins_per_branch=2,  # mean the action shape is 6, 2 discrete actions for each action dimension
            encoder_hidden_size_list=[256, 256, 128],
        ),
        learn=dict(
            batch_size=512,
            learning_rate=3e-4,
            ignore_done=True,
            target_update_freq=500,
            update_per_collect=20,
        ),
        collect=dict(
            n_sample=256,
            unroll_len=1,
        ),
        eval=dict(evaluator=dict(eval_freq=1000, )),
        other=dict(
            # Epsilon greedy with decay.
            eps=dict(
                # Decay type. Support ['exp', 'linear'].
                type='exp',
                start=1,
                end=0.05,
                decay=int(1e5),
            ),
            replay_buffer=dict(replay_buffer_size=int(1e6), )
        ),
    ),
)
halfcheetah_bdq_config = EasyDict(halfcheetah_bdq_config)
main_config = halfcheetah_bdq_config

halfcheetah_bdq_create_config = dict(
    env=dict(
        type='mujoco',
        import_names=['dizoo.mujoco.envs.mujoco_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='bdq', ),
)
halfcheetah_bdq_create_config = EasyDict(halfcheetah_bdq_create_config)
create_config = halfcheetah_bdq_create_config

if __name__ == "__main__":
    # or you can enter `ding -m serial_onpolicy -c halfcheetah_onbdq_config.py -s 0`
    from ding.entry import serial_pipeline
    serial_pipeline(
        (main_config, create_config),
        seed=0,
        max_env_step=10000000,
    )
