from easydict import EasyDict

qbert_pc_mcts_config = dict(
    exp_name='qbert_pc_mcts_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=1000000,
        env_id='Qbert-v4',
    ),
    policy=dict(
        cuda=True,
        expert_data_path='pong_expert/ez_pong_seed0.pkl',
        model=dict(
            obs_shape=[3, 96, 96],
            hidden_shape=[32, 8, 8],
            action_shape=6,
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
        type='atari',
        import_names=['dizoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='pc_mcts'),
)
qbert_pc_mcts_create_config = EasyDict(qbert_pc_mcts_create_config)
create_config = qbert_pc_mcts_create_config

if __name__ == "__main__":
    from ding.entry import serial_pipeline_pc_mcts
    serial_pipeline_pc_mcts([main_config, create_config], seed=0)
