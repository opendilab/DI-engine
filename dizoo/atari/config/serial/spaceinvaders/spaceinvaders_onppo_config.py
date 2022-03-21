from copy import deepcopy
from ding.entry import serial_pipeline_onpolicy
from easydict import EasyDict

spaceinvaders_ppo_config = dict(
    env=dict(
        collector_env_num=16,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=int(1e10),
        env_id='SpaceInvadersNoFrameskip-v4',
        frame_stack=4,
        manager=dict(shared_memory=False, )
    ),
    policy=dict(
        cuda=True,
        recompute_adv=True,
        action_space='discrete',
        model=dict(
            obs_shape=[4, 84, 84],
            action_shape=6,
            action_space='discrete',
            encoder_hidden_size_list=[64, 64, 128],
            actor_head_hidden_size=128,
            critic_head_hidden_size=128,
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
            ignore_done=False,
            grad_clip_type='clip_norm',
            grad_clip_value=0.5,
        ),
        collect=dict(
            n_sample=3200,
            unroll_len=1,
            discount_factor=0.99,
            gae_lambda=0.95,
        ),
        eval=dict(evaluator=dict(eval_freq=5000, )),
    ),
)
main_config = EasyDict(spaceinvaders_ppo_config)

spaceinvaders_ppo_create_config = dict(
    env=dict(
        type='atari',
        import_names=['dizoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='base'),
    # env_manager=dict(type='subprocess'),
    policy=dict(type='ppo'),
)
create_config = EasyDict(spaceinvaders_ppo_create_config)

# if __name__ == "__main__":
#     serial_pipeline_onpolicy([main_config, create_config], seed=0)


def train(args):
    main_config.exp_name = 'spaceinvaders_onppo_noig' + '_seed' + f'{args.seed}'
    import copy
    # 3125 iterations= 10M env steps / 3200
    serial_pipeline_onpolicy(
        [copy.deepcopy(main_config), copy.deepcopy(create_config)], seed=args.seed, max_iterations=3125
    )


if __name__ == "__main__":
    import argparse
    for seed in [0, 1, 2, 3, 4]:
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', '-s', type=int, default=seed)
        args = parser.parse_args()

        train(args)
