from easydict import EasyDict

qbert_trex_dqn_config = dict(
    exp_name='qbert_trex_dqn_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=30000,
        env_id='Qbert-v4',
        #'ALE/Qbert-v5' is available. But special setting is needed after gym make.
        frame_stack=4
    ),
    reward_model=dict(
        type='trex',
        min_snippet_length=30,
        max_snippet_length=100,
        checkpoint_min=0,
        checkpoint_max=100,
        checkpoint_step=100,
        learning_rate=1e-5,
        update_per_collect=1,
        expert_model_path='abs model path',
        reward_model_path='abs data path + ./qbert.params',
        offline_data_path='abs data path',
    ),
    policy=dict(
        cuda=True,
        priority=False,
        model=dict(
            obs_shape=[4, 84, 84],
            action_shape=6,
            encoder_hidden_size_list=[128, 128, 512],
        ),
        nstep=1,
        discount_factor=0.99,
        learn=dict(
            update_per_collect=10,
            batch_size=32,
            learning_rate=0.0001,
            target_update_freq=500,
        ),
        collect=dict(n_sample=100, ),
        eval=dict(evaluator=dict(eval_freq=4000, )),
        other=dict(
            eps=dict(
                type='exp',
                start=1.,
                end=0.05,
                decay=1000000,
            ),
            replay_buffer=dict(replay_buffer_size=400000, ),
        ),
    ),
)
qbert_trex_dqn_config = EasyDict(qbert_trex_dqn_config)
main_config = qbert_trex_dqn_config
qbert_trex_dqn_create_config = dict(
    env=dict(
        type='atari',
        import_names=['dizoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='dqn'),
)
qbert_trex_dqn_create_config = EasyDict(qbert_trex_dqn_create_config)
create_config = qbert_trex_dqn_create_config

if __name__ == "__main__":
    # Users should first run ``cartpole_dqn_config.py`` to save models (or checkpoints).
    # Note: Users should check that the checkpoints generated should include iteration_'checkpoint_min'.pth.tar,
    #   iteration_'checkpoint_max'.pth.tar with the interval checkpoint_step
    # where checkpoint_max, checkpoint_min, checkpoint_step are specified above.
    import argparse
    import torch
    from ding.entry import trex_collecting_data
    from ding.entry import serial_pipeline_reward_model_trex

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='please enter abs path for this file')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    # The function ``trex_collecting_data`` below is to collect episodic data for training the reward model in trex.
    trex_collecting_data(args)
    serial_pipeline_reward_model_trex((main_config, create_config))
