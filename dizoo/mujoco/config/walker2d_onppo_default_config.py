from easydict import EasyDict
from ding.entry import serial_pipeline_onpolicy

collector_env_num = 1
evaluator_env_num = 1
walker2d_ppo_default_config = dict(
    env=dict(
        env_id='Walker2d-v3',
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        use_act_scale=True,
        n_evaluator_episode=10,
        stop_value=6000,
    ),
    policy=dict(
        cuda=True,
        recompute_adv=True,
        action_space='continuous',
        model=dict(
            action_space='continuous',
            obs_shape=17,
            action_shape=6,
        ),
        learn=dict(
            epoch_per_collect=10,
            update_per_collect=1,
            batch_size=320,
            learning_rate=3e-4,
            value_weight=0.5,
            entropy_weight=0.001,
            clip_ratio=0.2,
            adv_norm=True,
            value_norm=True,
            # for onppo, when we recompute adv, we need the key done in data to split traj, so we must
            # use ignore_done=False here,
            # but when we add key traj_flag in data as the backup for key done, we could choose to use ignore_done=True
            # for halfcheetah, the length=1000
            # ignore_done=True,
            ignore_done=False,
            grad_clip_type='clip_norm',
            grad_clip_value=0.5,
        ),
        collect=dict(
            collector_env_num=collector_env_num,
            n_sample=3200,
            unroll_len=1,
            discount_factor=0.99,
            gae_lambda=0.95,
        ),
        eval=dict(evaluator=dict(eval_freq=500, )),
    ),
)
walker2d_ppo_default_config = EasyDict(walker2d_ppo_default_config)
main_config = walker2d_ppo_default_config

walker2d_ppo_create_default_config = dict(
    env=dict(
        type='mujoco',
        import_names=['dizoo.mujoco.envs.mujoco_env'],
    ),
    env_manager=dict(type='base'),
    # env_manager=dict(type='subprocess'),
    policy=dict(type='ppo', ),
)
walker2d_ppo_create_default_config = EasyDict(walker2d_ppo_create_default_config)
create_config = walker2d_ppo_create_default_config

# if __name__ == "__main__":
#     serial_pipeline_onpolicy([main_config, create_config], seed=0)


def train(args):
    main_config.exp_name = 'wlker2d_onppo_noig' + '_seed' + f'{args.seed}'
    import copy
    # 937.4 iterations= 3M env steps / 3200
    serial_pipeline_onpolicy(
        [copy.deepcopy(main_config), copy.deepcopy(create_config)], seed=args.seed, max_iterations=938
    )


if __name__ == "__main__":
    import argparse
    for seed in [0, 1, 2, 3, 4]:
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', '-s', type=int, default=seed)
        args = parser.parse_args()

        train(args)
