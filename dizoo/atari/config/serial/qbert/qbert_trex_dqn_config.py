from copy import deepcopy
from easydict import EasyDict

qbert_trex_dqn_config = dict(
    exp_name='qbert_trex_dqn',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=30000,
        env_id='QbertNoFrameskip-v4',
        frame_stack=4,
        manager=dict(shared_memory=False, )
    ),
    reward_model=dict(
        type='trex',
        algo_for_model='dqn',
        env_id='QbertNoFrameskip-v4',
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
