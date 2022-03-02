from easydict import EasyDict
from ding.entry import serial_pipeline

collector_env_num = 8
evaluator_env_num = 5
spaceinvaders_r2d2_config = dict(
    exp_name='spaceinvaders_r2d2_n5_bs20_ul80_rbs1e4',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=8,
        stop_value=int(1e6),
        env_id='SpaceInvadersNoFrameskip-v4',
        frame_stack=4,
        manager=dict(shared_memory=False, )
    ),
    policy=dict(
        cuda=True,
        priority=True,
        priority_IS_weight=True,
        model=dict(
            obs_shape=[4, 84, 84],
            action_shape=6,
            encoder_hidden_size_list=[128, 128, 512],
            res_link=False,
        ),
        discount_factor=0.997,
        burnin_step=20,
        nstep=5,
        # (int) the whole sequence length to unroll the RNN network minus
        # the timesteps of burnin part,
        # i.e., <the whole sequence length> = <burnin_step> + <unroll_len>
        unroll_len=80,
        learn=dict(
            # according to the R2D2 paper, actor parameter update interval is 400
            # environment timesteps, and in per collect phase, we collect 32 sequence
            # samples, the length of each samlpe sequence is <burnin_step> + <unroll_len>,
            # which is 100 in our seeting, 32*100/400=8, so we set update_per_collect=8
            # in most environments
            update_per_collect=8,
            batch_size=64,  # TODO(pu)
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
                replay_buffer_size=10000,  # TODO(pu)
                # (Float type) How much prioritization is used: 0 means no prioritization while 1 means full prioritization
                alpha=0.6,
                # (Float type)  How much correction is used: 0 means no correction while 1 means full correction
                beta=0.4,
            )
        ),
    ),
)
spaceinvaders_r2d2_config = EasyDict(spaceinvaders_r2d2_config)
main_config = spaceinvaders_r2d2_config
spaceinvaders_r2d2_create_config = dict(
    env=dict(
        type='atari',
        import_names=['dizoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='r2d2'),
)
spaceinvaders_r2d2_create_config = EasyDict(spaceinvaders_r2d2_create_config)
create_config = spaceinvaders_r2d2_create_config

# if __name__ == "__main__":
#     serial_pipeline([main_config, create_config], seed=0)


def train(args):
    main_config.exp_name = 'spaceinvaders_r2d2_n5_bs20_ul80_rbs1e4' + '_seed' + f'{args.seed}'
    import copy
    # 3125 iterations= 10M env steps / (100*32) env steps
    serial_pipeline(
        [copy.deepcopy(main_config), copy.deepcopy(create_config)],
        seed=args.seed,
        max_iterations=int(3125),
    )


if __name__ == "__main__":
    import argparse
    for seed in [0, 1, 2, 3, 4]:
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', '-s', type=int, default=seed)
        args = parser.parse_args()

        train(args)
