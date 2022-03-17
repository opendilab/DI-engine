from easydict import EasyDict
from ding.entry import serial_pipeline

nstep = 3
lunarlander_acer_default_config = dict(
    env=dict(
        # Whether to use shared memory. Only effective if "env_manager_type" is 'subprocess'
        manager=dict(shared_memory=True, ),
        # Env number respectively for collector and evaluator.
        collector_env_num=8,
        evaluator_env_num=5,
        env_id='LunarLander-v2',
        n_evaluator_episode=5,
        stop_value=200,
    ),
    policy=dict(
        # Whether to use cuda for network.
        cuda=False,
        # Model config used for model creating. Remember to change this,
        # especially "obs_shape" and "action_shape" according to specific env.
        model=dict(
            obs_shape=8,
            action_shape=4,
            encoder_hidden_size_list=[512, 64],
            # Whether to use dueling head.
        ),
        # Reward's future discount facotr, aka. gamma.
        discount_factor=0.99,
        # How many steps in td error.
        nstep=nstep,
        unroll_len=32,
        # learn_mode config
        learn=dict(
            # (int) collect n_sample data, train model update_per_collect times
            # here we follow impala serial pipeline
            update_per_collect=10,
            # (int) the number of data for a train iteration
            batch_size=32,
            # grad_clip_type='clip_norm',
            # clip_value=10,
            learning_rate_actor=0.0001,
            learning_rate_critic=0.0001,
            # (float) loss weight of the value network, the weight of policy network is set to 1
            # (float) loss weight of the entropy regularization, the weight of policy network is set to 1
            entropy_weight=0.0,
            # (float) discount factor for future reward, defaults int [0, 1]
            discount_factor=0.99,
            # (float) additional discounting parameter
            # (int) the trajectory length to calculate v-trace target
            # (float) clip ratio of importance weights
            c_clip_ratio=10,
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
        other=dict(replay_buffer=dict(replay_buffer_size=50000, ), ),
    ),
)
lunarlander_acer_default_config = EasyDict(lunarlander_acer_default_config)
main_config = lunarlander_acer_default_config

lunarlander_acer_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['dizoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='acer'),
    replay_buffer=dict(type='naive')
)
lunarlander_acer_create_config = EasyDict(lunarlander_acer_create_config)
create_config = lunarlander_acer_create_config

if __name__ == "__main__":
    serial_pipeline([main_config, create_config], seed=0)
