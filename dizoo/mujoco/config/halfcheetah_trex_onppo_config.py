from easydict import EasyDict

halfCheetah_trex_ppo_config = dict(
    exp_name='halfcheetah_trex_onppo_seed0',
    env=dict(
        env_id='HalfCheetah-v3',
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        collector_env_num=8,
        evaluator_env_num=10,
        n_evaluator_episode=10,
        stop_value=3000,
    ),
    reward_model=dict(
        min_snippet_length=30,
        max_snippet_length=100,
        checkpoint_min=10000,
        checkpoint_max=90000,
        checkpoint_step=10000,
        num_snippets=60000,
        learning_rate=1e-5,
        update_per_collect=1,
        # Users should add their own model path here. Model path should lead to a model.
        # Absolute path is recommended.
        # In DI-engine, it is ``exp_name/ckpt/ckpt_best.pth.tar``.
        # However, here in ``expert_model_path``, it is ``exp_name`` of the expert config.
        expert_model_path='model_path_placeholder',
        # Path where to store the reward model
        reward_model_path='data_path_placeholder + /HalfCheetah.params',
        # Users should add their own data path here. Data path should lead to a file to store data or load the stored data.
        # Absolute path is recommended.
        # In DI-engine, it is usually located in ``exp_name`` directory
        # See ding/entry/application_entry_trex_collect_data.py to collect the data
        data_path='data_path_placeholder',
    ),
    policy=dict(
        cuda=True,
        recompute_adv=True,
        model=dict(
            obs_shape=17,
            action_shape=6,
            action_space='continuous',
        ),
        action_space='continuous',
        learn=dict(
            epoch_per_collect=10,
            batch_size=64,
            learning_rate=3e-4,
            value_weight=0.5,
            entropy_weight=0.0,
            clip_ratio=0.2,
            adv_norm=True,
            value_norm=True,
            # for onppo, when we recompute adv, we need the key done in data to split traj, so we must
            # use ignore_done=False here,
            # but when we add key traj_flag in data as the backup for key done, we could choose to use ignore_done=True
            # for halfcheetah, the length=1000
            ignore_done=True,
            grad_clip_type='clip_norm',
            grad_clip_value=0.5,
        ),
        collect=dict(
            n_sample=2048,
            unroll_len=1,
            discount_factor=0.99,
            gae_lambda=0.97,
        ),
        eval=dict(evaluator=dict(eval_freq=5000, )),
    ),
)
halfCheetah_trex_ppo_config = EasyDict(halfCheetah_trex_ppo_config)
main_config = halfCheetah_trex_ppo_config

halfCheetah_trex_ppo_create_config = dict(
    env=dict(
        type='mujoco',
        import_names=['dizoo.mujoco.envs.mujoco_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='ppo', ),
    reward_model=dict(type='trex'),
)
halfCheetah_trex_ppo_create_config = EasyDict(halfCheetah_trex_ppo_create_config)
create_config = halfCheetah_trex_ppo_create_config

if __name__ == '__main__':
    # Users should first run ``halfcheetah_onppo_config.py`` to save models (or checkpoints).
    # Note: Users should check that the checkpoints generated should include iteration_'checkpoint_min'.pth.tar, iteration_'checkpoint_max'.pth.tar with the interval checkpoint_step
    # where checkpoint_max, checkpoint_min, checkpoint_step are specified above.
    import argparse
    import torch
    from ding.entry import trex_collecting_data
    from ding.entry import serial_pipeline_trex_onpolicy
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='please enter abs path for this file')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    # The function ``trex_collecting_data`` below is to collect episodic data for training the reward model in trex.
    trex_collecting_data(args)
    serial_pipeline_trex_onpolicy([main_config, create_config])
