from easydict import EasyDict

nstep = 3
halfcheetah_dqn_default_config = dict(
    exp_name='halfcheetah_dqn_seed0',
    env=dict(
        env_id='HalfCheetah-v3',
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        # (bool) Scale output action into legal range.
        use_act_scale=True,
        # Env number respectively for collector and evaluator.
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        # stop_value=3000,
        stop_value=int(1e6),  # max env steps
        each_dim_disc_size=3,  # n: discrete size of each dim in origin continuous action
    ),
    policy=dict(
        # Whether to use cuda for network.
        cuda=True,
        priority=False,
        # Reward's future discount factor, aka. gamma.
        discount_factor=0.99,
        # How many steps in td error.
        nstep=nstep,
        # learn_mode config
        # original_action_shape=3,  # m
        model=dict(
            obs_shape=17,
            action_shape=int(3**6),  # num of num_embeddings: K = n**m e.g. 2**6=64, 3**6=729, 4**6=4096
            # encoder_hidden_size_list=[128, 128, 64],  # small net
            # encoder_hidden_size_list=[256, 256, 128],  # middle net
            encoder_hidden_size_list=[512, 512, 256],  # large net
            # Whether to use dueling head.
            dueling=True,
        ),
        learn=dict(
            ignore_done=True,
            batch_size=512,
            learning_rate=3e-4,
            # Frequency of target network update.
            target_update_theta=0.001,
            update_per_collect=20,


            # rl_clip_grad=True,
            # grad_clip_type='clip_norm',
            # grad_clip_value=0.5,
        ),
        # collect_mode config
        collect=dict(
            # You can use either "n_sample" or "n_episode" in collector.collect.
            # Get "n_sample" samples per collect.
            n_sample=256,
            # Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
        ),
        eval=dict(evaluator=dict(eval_freq=1000, )),
        # command_mode config
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

halfcheetah_dqn_default_config = EasyDict(halfcheetah_dqn_default_config)
main_config = halfcheetah_dqn_default_config

halfcheetah_dqn_create_config = dict(
    env=dict(
        type='mujoco-disc',
        import_names=['dizoo.mujoco.envs.mujoco_env_disc'],
    ),
    env_manager=dict(type='subprocess'),
    # env_manager=dict(type='base'),
    policy=dict(type='dqn'),
)
halfcheetah_dqn_create_config = EasyDict(halfcheetah_dqn_create_config)
create_config = halfcheetah_dqn_create_config


def train(args):
    main_config.exp_name = 'data_halfcheetah/dqn_k64_largenet_upc20' + '_seed' + f'{args.seed}'+'_3M'
    serial_pipeline([copy.deepcopy(main_config), copy.deepcopy(create_config)], seed=args.seed, max_env_step=int(3e6))

if __name__ == "__main__":
    import copy
    import argparse
    from ding.entry import serial_pipeline

    # for seed in [0,1,2]:
    for seed in [0]:
        parser = argparse.ArgumentParser()
        parser.add_argument('--seed', '-s', type=int, default=seed)
        args = parser.parse_args()

        train(args)