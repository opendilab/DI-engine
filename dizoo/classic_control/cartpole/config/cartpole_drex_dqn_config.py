from easydict import EasyDict

cartpole_drex_dqn_config = dict(
    exp_name='cartpole_drex_dqn_seed0',
    env=dict(
        manager=dict(shared_memory=True, reset_inplace=True),
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=195,
    ),
    reward_model=dict(
        type='drex',
        exp_name='cartpole_drex_dqn_seed0',
        min_snippet_length=5,
        max_snippet_length=100,
        checkpoint_min=0,
        checkpoint_max=760,
        checkpoint_step=760,
        learning_rate=1e-5,
        update_per_collect=1,
        # path to expert models that generate demonstration data
        # Users should add their own model path here. Model path should lead to an exp_name.
        # Absolute path is recommended.
        # In DI-engine, it is ``exp_name``.
        # For example, if you want to use dqn to generate demos, you can use ``spaceinvaders_dqn``
        expert_model_path='cartpole_dqn_seed0/ckpt/ckpt_best.pth.tar',
        # path to save reward model
        # Users should add their own model path here.
        # Absolute path is recommended.
        # For example, if you use ``spaceinvaders_drex``, then the reward model will be saved in this directory.
        reward_model_path='cartpole_drex_dqn_seed0/cartpole.params',
        # path to save generated observations.
        # Users should add their own model path here.
        # Absolute path is recommended.
        # For example, if you use ``spaceinvaders_drex``, then all the generated data will be saved in this directory.
        offline_data_path='cartpole_drex_dqn_seed0',
        # path to pretrained bc model. If omitted, bc will be trained instead.
        # Users should add their own model path here. Model path should lead to a model ckpt.
        # Absolute path is recommended.
        # bc_path='bc_path_placeholder',
        # list of noises
        eps_list=[0, 0.5, 1],
        num_trajs_per_bin=20,
        num_trajs=6,
        num_snippets=6000,
        bc_iterations=6000,
        hidden_size_list=[512, 64, 1],
        obs_shape=4,
        action_shape=2,
    ),
    policy=dict(
        cuda=False,
        model=dict(
            obs_shape=4,
            action_shape=2,
            encoder_hidden_size_list=[128, 128, 64],
            dueling=True,
        ),
        nstep=1,
        discount_factor=0.97,
        learn=dict(
            batch_size=64,
            learning_rate=0.001,
        ),
        collect=dict(
            n_sample=8,
            collector=dict(
                get_train_sample=False,
                reward_shaping=False,
            ),
        ),
        eval=dict(evaluator=dict(eval_freq=40, )),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.1,
                decay=10000,
            ),
            replay_buffer=dict(replay_buffer_size=200000, ),
        ),
    ),
)
cartpole_drex_dqn_config = EasyDict(cartpole_drex_dqn_config)
main_config = cartpole_drex_dqn_config
cartpole_drex_dqn_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='dqn'),
)
cartpole_drex_dqn_create_config = EasyDict(cartpole_drex_dqn_create_config)
create_config = cartpole_drex_dqn_create_config

if __name__ == "__main__":
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
    args.cfg[0].policy.collect.n_episode = 8
    del args.cfg[0].policy.collect.n_sample
    drex_collecting_data(args)
    serial_pipeline_reward_model_offpolicy((main_config, create_config), pretrain_reward=True, cooptrain_reward=False)
