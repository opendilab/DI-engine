from copy import deepcopy
from ding.entry import serial_pipeline
from easydict import EasyDict

freeway_a2c_config = dict(
    env=dict(
        collector_env_num=16,
        evaluator_env_num=4,
        n_evaluator_episode=8,
        stop_value=20,
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
            update_per_collect=1,
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
            # (int) collect n_sample data, train model n_iteration times
            n_sample=160,
            # (float) the trade-off factor lambda to balance 1step td and mc
            gae_lambda=0.99,
            discount_factor=0.99,
        ),
        eval=dict(evaluator=dict(eval_freq=100, )),
        other=dict(replay_buffer=dict(
            replay_buffer_size=160,
            max_use=1,
        ), ),
    ),
)
main_config = EasyDict(freeway_a2c_config)

freeway_a2c_create_config = dict(
    env=dict(
        type='atari',
        import_names=['dizoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='a2c'),
)
create_config = EasyDict(freeway_a2c_create_config)

if __name__ == '__main__':
    serial_pipeline((main_config, create_config), seed=0)
