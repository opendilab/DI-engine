from easydict import EasyDict

cartpole_trex_dqn_config = dict(
    exp_name='cartpole_trex_dqn_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=195,
    ),
    reward_model=dict(
        type='trex',
        exp_name='cartpole_trex_dqn_seed0',
        min_snippet_length=5,
        max_snippet_length=100,
        checkpoint_min=0,
        checkpoint_max=500,
        checkpoint_step=100,
        learning_rate=1e-5,
        update_per_collect=1,
        num_trajs=6,
        num_snippets=6000,
        # Users should add their own model path here. Model path should lead to a model.
        # Absolute path is recommended.
        # In DI-engine, it is ``exp_name``.
        expert_model_path='cartpole_dqn_seed0',  # expert model experiment directory path
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
            update_per_collect=5,
            batch_size=64,
            learning_rate=0.001,
        ),
        collect=dict(n_sample=8),
        eval=dict(evaluator=dict(eval_freq=40, )),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.1,
                decay=10000,
            ),
            replay_buffer=dict(replay_buffer_size=20000, ),
        ),
    ),
)
cartpole_trex_dqn_config = EasyDict(cartpole_trex_dqn_config)
main_config = cartpole_trex_dqn_config
cartpole_trex_dqn_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='dqn'),
    reward_model=dict(type='trex'),
)
cartpole_trex_dqn_create_config = EasyDict(cartpole_trex_dqn_create_config)
create_config = cartpole_trex_dqn_create_config

if __name__ == "__main__":
    # Users should first run ``cartpole_dqn_config.py`` to save models (or checkpoints).
    # Note: Users should check that the checkpoints generated should include iteration_'checkpoint_min'.pth.tar, iteration_'checkpoint_max'.pth.tar with the interval checkpoint_step
    # where checkpoint_max, checkpoint_min, checkpoint_step are specified above.
    # example of running this file:
    # python cartpole_trex_dqn_config.py --cfg cartpole_trex_dqn_config.py --seed 0 --device cpu
    import argparse
    import torch
    from ding.entry import trex_collecting_data
    from ding.entry import serial_pipeline_reward_model_offpolicy
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='please enter abs path for this file')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    # The function ``trex_collecting_data`` below is to collect episodic data for training the reward model in trex.
    trex_collecting_data(args)
    serial_pipeline_reward_model_offpolicy((main_config, create_config), pretrain_reward_model=True, cooptrain_reward_model=False)
