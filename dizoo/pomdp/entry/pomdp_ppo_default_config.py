from ding.entry import serial_pipeline
from easydict import EasyDict

pong_ppo_config = dict(
    env=dict(
        collector_env_num=16,
        evaluator_env_num=4,
        n_evaluator_episode=8,
        stop_value=20,
        env_id='Pong-ramNoFrameskip-v4',
        frame_stack=4,
        warp_frame=False,
        use_ram=True,
        pomdp=dict(noise_scale=0.01, zero_p=0.2, reward_noise=0.01, duplicate_p=0.2),
        manager=dict(shared_memory=False, )
    ),
    policy=dict(
        cuda=True,
        # (bool) whether use on-policy training pipeline(behaviour policy and training policy are the same)
        model=dict(
            obs_shape=[
                512,
            ],
            action_shape=6,
            encoder_hidden_size_list=[512, 512, 256],
            actor_head_hidden_size=256,
            actor_head_layer_num=2,
            critic_head_hidden_size=256,
            critic_head_layer_num=2,
        ),
        learn=dict(
            update_per_collect=16,
            batch_size=128,
            # (bool) Whether to normalize advantage. Default to False.
            normalize_advantage=False,
            learning_rate=0.0001,
            weight_decay=0,
            # (float) loss weight of the value network, the weight of policy network is set to 1
            value_weight=0.5,
            # (float) loss weight of the entropy regularization, the weight of policy network is set to 1
            entropy_weight=0.03,
            clip_ratio=0.1,
        ),
        collect=dict(
            # (int) collect n_sample data, train model n_iteration times
            n_sample=1024,
            # (float) the trade-off factor lambda to balance 1step td and mc
            gae_lambda=0.97,
            discount_factor=0.99,
        ),
        eval=dict(evaluator=dict(eval_freq=200, )),
        other=dict(replay_buffer=dict(
            replay_buffer_size=100000,
            max_use=3,
            min_sample_ratio=1,
        ), ),
    ),
)
main_config = EasyDict(pong_ppo_config)

pong_ppo_create_config = dict(
    env=dict(
        type='pomdp',
        import_names=['app_zoo.pomdp.envs.atari_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='ppo'),
)
create_config = EasyDict(pong_ppo_create_config)

if __name__ == '__main__':
    serial_pipeline((main_config, create_config), seed=0)
