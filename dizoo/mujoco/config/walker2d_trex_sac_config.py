from easydict import EasyDict

walker2d_trex_sac_config = dict(
    exp_name='walker2d_trex_sac_seed0',
    env=dict(
        env_id='Walker2d-v3',
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        collector_env_num=1,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=6000,
    ),
    reward_model=dict(
        learning_rate=1e-5,
        min_snippet_length=30,
        max_snippet_length=100,
        checkpoint_min=1000,
        checkpoint_max=9000,
        checkpoint_step=1000,
        update_per_collect=1,
        # Users should add their own model path here. Model path should lead to a model.
        # Absolute path is recommended.
        # In DI-engine, it is ``exp_name/ckpt/ckpt_best.pth.tar``.
        # However, here in ``expert_model_path``, it is ``exp_name`` of the expert config.
        expert_model_path='model_path_placeholder',
        # Path where to store the reward model
        reward_model_path='data_path_placeholder + /Walker2d.params',
        # Users should add their own data path here. Data path should lead to a file to store data or load the stored data.
        # Absolute path is recommended.
        # In DI-engine, it is usually located in ``exp_name`` directory
        # See ding/entry/application_entry_trex_collect_data.py to collect the data
        data_path='data_path_placeholder',
    ),
    policy=dict(
        cuda=True,
        random_collect_size=10000,
        model=dict(
            obs_shape=17,
            action_shape=6,
            twin_critic=True,
            action_space='reparameterization',
            actor_head_hidden_size=256,
            critic_head_hidden_size=256,
        ),
        learn=dict(
            update_per_collect=1,
            batch_size=256,
            learning_rate_q=1e-3,
            learning_rate_policy=1e-3,
            learning_rate_alpha=3e-4,
            ignore_done=False,
            target_theta=0.005,
            discount_factor=0.99,
            alpha=0.2,
            reparameterization=True,
            auto_alpha=False,
        ),
        collect=dict(
            n_sample=1,
            unroll_len=1,
        ),
        command=dict(),
        eval=dict(),
        other=dict(replay_buffer=dict(replay_buffer_size=1000000, ), ),
    ),
)

walker2d_trex_sac_config = EasyDict(walker2d_trex_sac_config)
main_config = walker2d_trex_sac_config

walker2d_trex_sac_create_config = dict(
    env=dict(
        type='mujoco',
        import_names=['dizoo.mujoco.envs.mujoco_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(
        type='sac',
        import_names=['ding.policy.sac'],
    ),
    replay_buffer=dict(type='naive', ),
)
walker2d_trex_sac_create_config = EasyDict(walker2d_trex_sac_create_config)
create_config = walker2d_trex_sac_create_config

if __name__ == '__main__':
    # Users should first run ``walker2d_sac_config.py`` to save models (or checkpoints).
    # Note: Users should check that the checkpoints generated should include iteration_'checkpoint_min'.pth.tar, iteration_'checkpoint_max'.pth.tar with the interval checkpoint_step
    # where checkpoint_max, checkpoint_min, checkpoint_step are specified above.
    import argparse
    import torch
    from ding.entry import trex_collecting_data
    from ding.entry import serial_pipeline_trex
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='please enter abs path for this file')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    # The function ``trex_collecting_data`` below is to collect episodic data for training the reward model in trex.
    trex_collecting_data(args)
    serial_pipeline_trex([main_config, create_config])
