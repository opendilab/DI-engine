from easydict import EasyDict

seq_len = 4
qbert_pc_mcts_config = dict(
    exp_name='pong_pc_mcts_seed0',
    env=dict(
        manager=dict(
            episode_num=float('inf'),
            max_retry=5,
            step_timeout=None,
            auto_reset=True,
            reset_timeout=None,
            retry_type='reset',
            retry_waiting_time=0.1,
            shared_memory=False,
            copy_on_get=True,
            context='fork',
            wait_num=float('inf'),
            step_wait_timeout=None,
            connect_timeout=60,
            reset_inplace=False,
            cfg_type='SyncSubprocessEnvManagerDict',
            type='subprocess',
        ),
        dqn_expert_data=False,
        cfg_type='AtariLightZeroEnvDict',
        collector_env_num=8,
        evaluator_env_num=3,
        n_evaluator_episode=3,
        env_name='PongNoFrameskip-v4',
        stop_value=20,
        collect_max_episode_steps=10800,
        eval_max_episode_steps=108000,
        frame_skip=4,
        obs_shape=[12, 96, 96],
        episode_life=True,
        gray_scale=False,
        cvt_string=False,
        game_wrapper=True,
    ),
    policy=dict(
        cuda=True,
        expert_data_path='pong-v4-expert.pkl',
        seq_len=seq_len,
        seq_action=True,
        mask_seq_action=False,
        model=dict(
            obs_shape=[3, 96, 96],
            hidden_shape=[64, 6, 6],
            action_dim=6,
            seq_len=seq_len,
        ),
        learn=dict(
            batch_size=64,
            learning_rate=0.01,
            learner=dict(hook=dict(save_ckpt_after_iter=1000)),
            train_epoch=20,
        ),
        eval=dict(evaluator=dict(eval_freq=40, ))
    ),
)
qbert_pc_mcts_config = EasyDict(qbert_pc_mcts_config)
main_config = qbert_pc_mcts_config
qbert_pc_mcts_create_config = dict(
    env=dict(
        type='atari_lightzero',
        import_names=['zoo.atari.envs.atari_lightzero_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='pc_mcts'),
)
qbert_pc_mcts_create_config = EasyDict(qbert_pc_mcts_create_config)
create_config = qbert_pc_mcts_create_config

if __name__ == "__main__":
    from ding.entry import serial_pipeline_pc_mcts
    serial_pipeline_pc_mcts([main_config, create_config], seed=0)
