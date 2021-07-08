from easydict import EasyDict
from ding.entry import serial_pipeline

nstep = 3
lunarlander_dqn_default_config = dict(
    env=dict(
        # Whether to use shared memory. Only effective if "env_manager_type" is 'subprocess'
        manager=dict(shared_memory=True, ),
        # Env number respectively for collector and evaluator.
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=5,
    ),
    policy=dict(
        # Whether to use cuda for network.
        cuda=False,
        # Whether the RL algorithm is on-policy or off-policy.
        on_policy=False,
        # Model config used for model creating. Remember to change this, especially "obs_dim" and "action_dim" according to specific env.
        model=dict(
            obs_dim=8,
            action_dim=4,
            encoder_hidden_dim_list=[512, 64],
            # Whether to use dueling head.
            dueling=True,
        ),
        # Reward's future discount facotr, aka. gamma.
        discount_factor=0.99,
        # How many steps in td error.
        nstep=nstep,
        # learn_mode config
        learn=dict(
            # How many steps to train after collector's one collection. Bigger "train_iteration" means bigger off-policy.
            # collect data -> train fixed steps -> collect data -> ...
            update_per_collect=10,
            batch_size=64,
            learning_rate=0.001,
            # Frequence of target network update.
            target_update_freq=100,
        ),
        # collect_mode config
        collect=dict(
            # You can use either "n_sample" or "n_episode" in collector.collect.
            # Get "n_sample" samples per collect.
            n_sample=64,
            # Cut trajectories into pieces with length "unrol_len".
            unroll_len=1,
        ),
        # command_mode config
        other=dict(
            # Epsilon greedy with decay.
            eps=dict(
                # Decay type. Support ['exp', 'linear'].
                type='exp',
                start=0.95,
                end=0.1,
                decay=50_000,
            ),
            replay_buffer=dict(
                replay_buffer_size=100000,
            )
        ),
    ),
)
lunarlander_dqn_default_config = EasyDict(lunarlander_dqn_default_config)
main_config = lunarlander_dqn_default_config

lunarlander_dqn_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['dizoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='dqn'),
)
lunarlander_dqn_create_config = EasyDict(lunarlander_dqn_create_config)
create_config = lunarlander_dqn_create_config

if __name__ == "__main__":
    serial_pipeline([main_config, create_config], seed=0)