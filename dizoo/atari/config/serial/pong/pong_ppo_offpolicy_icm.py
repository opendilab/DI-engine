from copy import deepcopy
from ding.entry import serial_pipeline_reward_model
from easydict import EasyDict

pong_ppo_icm_config = dict(
    exp_name='pong_ppo_icm',
    env=dict(
        collector_env_num=16,
        evaluator_env_num=4,
        n_evaluator_episode=8,
        stop_value=20,
        env_id='PongNoFrameskip-v4',
        frame_stack=4,
    ),
    reward_model=dict(
        intrinsic_reward_type='add',
        learning_rate=0.001,
        obs_shape=[4, 84, 84],
        action_shape=6,
        batch_size=32,
        update_per_collect=10,
    ),
    policy=dict(
        cuda=True,
        on_policy=False,
        # (bool) whether use on-policy training pipeline(behaviour policy and training policy are the same)
        model=dict(
            obs_shape=[4, 84, 84],
            action_shape=6,
            encoder_hidden_size_list=[64, 64, 128],
            actor_head_hidden_size=128,
            critic_head_hidden_size=128,
        ),
        learn=dict(
            update_per_collect=24,
            batch_size=128,
            # (bool) Whether to normalize advantage. Default to False.
            adv_norm=False,
            learning_rate=0.0002,
            # (float) loss weight of the value network, the weight of policy network is set to 1
            value_weight=0.5,
            # (float) loss weight of the entropy regularization, the weight of policy network is set to 1
            entropy_weight=0.015,
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
        other=dict(replay_buffer=dict(
            replay_buffer_size=100000,
            max_use=3,
        ), ),
    ),
)
main_config = EasyDict(pong_ppo_icm_config)
pong_ppo_create_config = dict(
    env=dict(
        type='atari',
        import_names=['dizoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='ppo_offpolicy'),
    reward_model=dict(type='icm'),
)
create_config = EasyDict(pong_ppo_create_config)

if __name__ == '__main__':
    serial_pipeline_reward_model([main_config, create_config], seed=0)
