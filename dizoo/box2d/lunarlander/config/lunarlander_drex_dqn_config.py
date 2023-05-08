from easydict import EasyDict

nstep = 1
lunarlander_drex_dqn_config = dict(
    exp_name='lunarlander_drex_dqn_seed0',
    env=dict(
        # Whether to use shared memory. Only effective if "env_manager_type" is 'subprocess'
        # Env number respectively for collector and evaluator.
        collector_env_num=8,
        evaluator_env_num=8,
        env_id='LunarLander-v2',
        n_evaluator_episode=8,
        stop_value=200,
    ),
    reward_model=dict(
        type='drex',
        exp_name='lunarlander_drex_dqn_seed0',
        min_snippet_length=30,
        max_snippet_length=100,
        checkpoint_min=1000,
        checkpoint_max=9000,
        checkpoint_step=1000,
        num_snippets=60000,
        num_trajs_per_bin=20,
        num_trajs=6,
        bc_iterations=6000,
        learning_rate=1e-5,
        update_per_collect=1,
        # Users should add their own model path here. Model path should lead to a model.
        # Absolute path is recommended.
        # In DI-engine, it is ``exp_name/ckpt/ckpt_best.pth.tar``.
        expert_model_path='lunarlander_dqn_seed0/ckpt/ckpt_best.pth.tar',
        reward_model_path='lunarlander_dqn_seed0/cartpole.params',
        offline_data_path='lunarlander_drex_dqn_seed0',
        hidden_size_list=[512, 64, 1],
        obs_shape=8,
        action_shape=4,
        eps_list=[0, 0.5, 1],
    ),
    policy=dict(
        # Whether to use cuda for network.
        cuda=False,
        model=dict(
            obs_shape=8,
            action_shape=4,
            encoder_hidden_size_list=[512, 64],
            # Whether to use dueling head.
            dueling=True,
        ),
        # Reward's future discount factor, aka. gamma.
        discount_factor=0.99,
        # How many steps in td error.
        nstep=nstep,
        # learn_mode config
        learn=dict(
            update_per_collect=10,
            batch_size=64,
            learning_rate=0.001,
            # Frequency of target network update.
            target_update_freq=100,
        ),
        # collect_mode config
        collect=dict(
            # You can use either "n_sample" or "n_episode" in collector.collect.
            # Get "n_sample" samples per collect.
            n_sample=64,
            # Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
            collector=dict(get_train_sample=False, reward_shaping=False,), 
        ),
        # command_mode config
        other=dict(
            # Epsilon greedy with decay.
            eps=dict(
                # Decay type. Support ['exp', 'linear'].
                type='exp',
                start=0.95,
                end=0.1,
                decay=50000,
            ),
            replay_buffer=dict(replay_buffer_size=100000, )
        ),
    ),
)
lunarlander_drex_dqn_config = EasyDict(lunarlander_drex_dqn_config)
main_config = lunarlander_drex_dqn_config

lunarlander_drex_dqn_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['dizoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='dqn'),
    reward_model=dict(type='drex'),
)
lunarlander_drex_dqn_create_config = EasyDict(lunarlander_drex_dqn_create_config)
create_config = lunarlander_drex_dqn_create_config

if __name__ == '__main__':
    # Users should first run ``lunarlander_dqn_config.py`` to save models (or checkpoints).
    # Note: Users should check that the checkpoints generated should include iteration_'checkpoint_min'.pth.tar, iteration_'checkpoint_max'.pth.tar with the interval checkpoint_step
    # where checkpoint_max, checkpoint_min, checkpoint_step are specified above.
    import argparse
    import torch
    from ding.config import read_config
    from ding.entry import drex_collecting_data
    from ding.entry import serial_pipeline_reward_model_offpolicy
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='please enter abs path for this file')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    args.cfg = read_config(args.cfg)
    args.cfg[1].policy.type = 'bc'
    args.cfg[0].policy.collect.n_episode = 64
    del args.cfg[0].policy.collect.n_sample
    drex_collecting_data(args)
    serial_pipeline_reward_model_offpolicy((main_config, create_config), pretrain_reward=True, cooptrain_reward=False, max_env_step=int(1e7))
