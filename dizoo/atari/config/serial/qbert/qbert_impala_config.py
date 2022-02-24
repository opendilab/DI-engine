from copy import deepcopy
from easydict import EasyDict

qbert_impala_config = dict(
    exp_name='qbert_impala_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=4,
        n_evaluator_episode=8,
        stop_value=10000000000,
        env_id='QbertNoFrameskip-v4',
        frame_stack=4,
        manager=dict(shared_memory=False, )
    ),
    policy=dict(
        cuda=True,
        # (int) the trajectory length to calculate v-trace target
        unroll_len=32,
        model=dict(
            obs_shape=[4, 84, 84],
            action_shape=6,
            encoder_hidden_size_list=[128, 128, 256, 512],
            critic_head_hidden_size=512,
            critic_head_layer_num=3,
            actor_head_hidden_size=512,
            actor_head_layer_num=3,
        ),
        learn=dict(
            # (int) collect n_sample data, train model update_per_collect times
            # here we follow impala serial pipeline
            update_per_collect=10,  # update_per_collect show be in [1, 10]
            # (int) the number of data for a train iteration
            batch_size=128,
            grad_clip_type='clip_norm',
            clip_value=5,
            learning_rate=0.0003,
            # (float) loss weight of the value network, the weight of policy network is set to 1
            value_weight=0.5,
            # (float) loss weight of the entropy regularization, the weight of policy network is set to 1
            entropy_weight=0.01,
            # (float) discount factor for future reward, defaults int [0, 1]
            discount_factor=0.99,
            # (float) additional discounting parameter
            lambda_=0.95,
            # (float) clip ratio of importance weights
            rho_clip_ratio=1.0,
            # (float) clip ratio of importance weights
            c_clip_ratio=1.0,
            # (float) clip ratio of importance sampling
            rho_pg_clip_ratio=1.0,
        ),
        collect=dict(
            # (int) collect n_sample data, train model n_iteration times
            n_sample=16,
            collector=dict(collect_print_freq=1000, ),
        ),
        eval=dict(evaluator=dict(eval_freq=5000, )),
        other=dict(replay_buffer=dict(replay_buffer_size=10000, ), ),
    ),
)
main_config = EasyDict(qbert_impala_config)

qbert_impala_create_config = dict(
    env=dict(
        type='atari',
        import_names=['dizoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='impala'),
    replay_buffer=dict(type='naive'),
)
create_config = EasyDict(qbert_impala_create_config)

if __name__ == '__main__':
    from ding.entry import serial_pipeline
    serial_pipeline((main_config, create_config), seed=0)
