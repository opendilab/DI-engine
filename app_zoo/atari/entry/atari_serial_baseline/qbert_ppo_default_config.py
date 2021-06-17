from copy import deepcopy
from nervex.entry import serial_pipeline
from easydict import EasyDict

qbert_ppo_config = dict(
    env=dict(
        collector_env_num=16,
        evaluator_env_num=4,
        n_evaluator_episode=8,
        stop_value=1000000,
        env_id='QbertNoFrameskip-v4',
        frame_stack=4,
        manager=dict(
            shared_memory=False,
        )
    ),
    policy=dict(
        cuda=True,
        on_policy=False,
        # (bool) whether use on-policy training pipeline(behaviour policy and training policy are the same)
        model=dict(
            model_type='conv_vac',
            import_names=['nervex.model.vac'],
            obs_shape=[4, 84, 84],
            action_shape=6,
            embedding_size=128,
        ),
        learn=dict(
            update_per_collect=24,
            batch_size=128,
            # (bool) Whether to normalize advantage. Default to False.
            normalize_advantage=False,
            learning_rate=0.0001,
            weight_decay=0,
            # (float) loss weight of the value network, the weight of policy network is set to 1
            value_weight=1.0,
            # (float) loss weight of the entropy regularization, the weight of policy network is set to 1
            entropy_weight=0.03,
            clip_ratio=0.1,
        ),
        collect=dict(
            # (int) collect n_sample data, train model n_iteration times
            n_sample=1024,
            # (float) the trade-off factor lambda to balance 1step td and mc
            gae_lambda=0.95,
            discount_factor=0.99,
        ),
        eval=dict(evaluator=dict(eval_freq=1000, )),
        other=dict(
            replay_buffer=dict(
                replay_buffer_size=100000,
                max_use=3,
                min_sample_ratio=1,
            ),
        ),
    ),
)
main_config = EasyDict(qbert_ppo_config)

qbert_ppo_create_config = dict(
    env=dict(
        type='atari',
        import_names=['app_zoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='ppo'),
)
create_config = EasyDict(qbert_ppo_create_config)

if __name__ == '__main__':
    serial_pipeline((main_config, create_config), seed=0)
