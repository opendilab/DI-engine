from copy import deepcopy
from ding.entry import serial_pipeline
from easydict import EasyDict

enduro_impala_config = dict(
    env=dict(
        collector_env_num=16,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=10000000000,
        env_id='EnduroNoFrameskip-v4',
        frame_stack=4,
        manager=dict(shared_memory=False, )
    ),
    policy=dict(
        cuda=True,
        # (int) the trajectory length to calculate v-trace target
        unroll_len=64,
        model=dict(
            obs_shape=[4, 84, 84],
            action_shape=9,
            encoder_hidden_size_list=[128, 128, 512],
            critic_head_hidden_size=512,
            critic_head_layer_num=2,
            actor_head_hidden_size=512,
            actor_head_layer_num=2,
        ),
        learn=dict(
            # (int) collect n_sample data, train model update_per_collect times
            # here we follow ppo serial pipeline
            update_per_collect=10,
            # (int) the number of data for a train iteration
            batch_size=128,
            grad_clip_type='clip_norm',
            clip_value=10.0,
            learning_rate=0.0001,
            # (float) loss weight of the value network, the weight of policy network is set to 1
            value_weight=1.0,
            # (float) loss weight of the entropy regularization, the weight of policy network is set to 1
            entropy_weight=0.0000001,
            # (float) discount factor for future reward, defaults int [0, 1]
            discount_factor=0.99,
            # (float) additional discounting parameter
            lambda_=1.0,
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
            # (float) discount factor for future reward, defaults int [0, 1]
            discount_factor=0.99,
            gae_lambda=0.95,
            collector=dict(collect_print_freq=1000, ),
        ),
        eval=dict(evaluator=dict(eval_freq=5000, )),
        other=dict(replay_buffer=dict(
            type='naive',
            replay_buffer_size=500000,
            max_use=100,
        ), ),
    ),
)
main_config = EasyDict(enduro_impala_config)

enduro_impala_create_config = dict(
    env=dict(
        type='atari',
        import_names=['dizoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='impala'),
)
create_config = EasyDict(enduro_impala_create_config)

if __name__ == '__main__':
    from ding.entry import serial_pipeline

    serial_pipeline((main_config, create_config), seed=0)
