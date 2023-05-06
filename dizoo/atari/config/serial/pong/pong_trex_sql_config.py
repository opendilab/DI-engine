from easydict import EasyDict

pong_trex_sql_config = dict(
    exp_name='pong_trex_sql_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=20,
        env_id='PongNoFrameskip-v4',
        #'ALE/Pong-v5' is available. But special setting is needed after gym make.
        frame_stack=4,
    ),
    reward_model=dict(
        type='trex',
        exp_name='pong_trex_sql_seed0',
        min_snippet_length=50,
        max_snippet_length=100,
        checkpoint_min=10000,
        checkpoint_max=50000,
        checkpoint_step=10000,
        learning_rate=1e-5,
        update_per_collect=1,
        # Users should add their own model path here. Model path should lead to a model.
        # Absolute path is recommended.
        # In DI-engine, it is ``exp_name/ckpt/ckpt_best.pth.tar``.
        # However, here in ``expert_model_path``, it is ``exp_name`` of the expert config.
        expert_model_path='pong_sql_seed0',
        hidden_size_list=[512, 64, 1],
        obs_shape=[4, 84, 84],
        action_shape=6,
    ),
    policy=dict(
        cuda=False,
        priority=False,
        model=dict(
            obs_shape=[4, 84, 84],
            action_shape=6,
            encoder_hidden_size_list=[128, 128, 512],
        ),
        nstep=1,
        discount_factor=0.99,
        learn=dict(update_per_collect=10, batch_size=32, learning_rate=0.0001, target_update_freq=500, alpha=0.12),
        collect=dict(n_sample=96, ),
        other=dict(
            eps=dict(
                type='exp',
                start=1.,
                end=0.05,
                decay=250000,
            ),
            replay_buffer=dict(replay_buffer_size=100000, ),
        ),
    ),
)
pong_trex_sql_config = EasyDict(pong_trex_sql_config)
main_config = pong_trex_sql_config
pong_trex_sql_create_config = dict(
    env=dict(
        type='atari',
        import_names=['dizoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='sql'),
    reward_model=dict(type='trex'),
)
pong_trex_sql_create_config = EasyDict(pong_trex_sql_create_config)
create_config = pong_trex_sql_create_config

if __name__ == '__main__':
    # Users should first run ``ppo_sql_config.py`` to save models (or checkpoints).
    # Note: Users should check that the checkpoints generated should include iteration_'checkpoint_min'.pth.tar, iteration_'checkpoint_max'.pth.tar with the interval checkpoint_step
    # where checkpoint_max, checkpoint_min, checkpoint_step are specified above.
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
    serial_pipeline_reward_model_offpolicy((main_config, create_config), pretrain_reward=True, cooptrain_reward=False)
