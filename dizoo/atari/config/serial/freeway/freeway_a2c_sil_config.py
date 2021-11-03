from copy import deepcopy
from ding.entry import serial_pipeline, serial_pipeline_sil
from easydict import EasyDict

freeway_a2c_config = dict(
    env=dict(
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=10000,
        env_id='FreewayNoFrameskip-v4',
        frame_stack=4,
        manager=dict(shared_memory=False, )
    ),
    policy=dict(
        cuda=False,
        on_policy=True,
        # (bool) whether use on-policy training pipeline(behaviour policy and training policy are the same)
        model=dict(
            obs_shape=[4, 84, 84],
            action_shape=3,
            encoder_hidden_size_list=[64, 64, 128],
            actor_head_hidden_size=128,
            critic_head_hidden_size=128,
        ),
        learn=dict(
            update_per_collect=50,
            batch_size=160,
            # (bool) Whether to normalize advantage. Default to False.
            adv_norm=False,
            learning_rate=0.0001,
            # (float) loss weight of the value network, the weight of policy network is set to 1
            value_weight=0.5,
            # (float) loss weight of the entropy regularization, the weight of policy network is set to 1
            entropy_weight=0.01,
            grad_norm=0.5,
            betas=(0.9, 0.999),
        ),
        collect=dict(
            collector=dict(
                type='episode',
                get_train_sample=True,
            ),
            # (int) collect n_sample data, train model n_iteration times
            n_episode=8,
            # (float) the trade-off factor lambda to balance 1step td and mc
            gae_lambda=0.95,
        ),
        eval=dict(evaluator=dict(eval_freq=500, )),
        other=dict(sil=dict(
            value_weight=0.5,
            learning_rate=0.001,
        ),
            replay_buffer=dict(type='naive',replay_buffer_size=500000, ),
        ),
    ),
)
freeway_a2c_config = EasyDict(freeway_a2c_config)
main_config = freeway_a2c_config
freeway_a2c_create_config = dict(
    env=dict(
        type='atari',
        import_names=['dizoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='a2c'),
)
freeway_a2c_create_config = EasyDict(freeway_a2c_create_config)
create_config = freeway_a2c_create_config

if __name__ == '__main__':
    serial_pipeline_sil((main_config, create_config), seed=0)
