from easydict import EasyDict
from ding.entry import serial_pipeline_gail
from lunarlander_dqn_config import lunarlander_dqn_default_config, lunarlander_dqn_create_config

nstep = 1
lunarlander_dqn_gail_default_config = dict(
    exp_name='lunarlander_dqn_gail',
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
    reward_model=dict(
        type='gail',
        input_size=9,
        hidden_size=64,
        batch_size=64,
        learning_rate=1e-3,
        update_per_collect=100,
        expert_data_path='lunarlander_dqn/expert_data.pkl',  # path where the expert data is stored
        expert_load_path='lunarlander_dqn/ckpt/ckpt_best.pth.tar',  # path to the expert state_dict
        collect_count=100000,
        load_path='lunarlander_dqn_gail/reward_model/ckpt/ckpt_last.pth.tar',
    ),
    policy=dict(
        load_path='lunarlander_dqn_gail/ckpt/ckpt_best.pth.tar',
        # Whether to use cuda for network.
        cuda=False,
        # Whether the RL algorithm is on-policy or off-policy.
        on_policy=False,
        model=dict(
            obs_shape=8,
            action_shape=4,
            encoder_hidden_size_list=[512, 64],
            # Whether to use dueling head.
            dueling=True,
        ),
        # Reward's future discount factor, aka. gamma.
        discount_factor=0.99,
        # How many steps in td error.
        nstep=nstep,
        # learn_mode config
        learn=dict(
            update_per_collect=10,
            batch_size=64,
            learning_rate=0.001,
            # Frequency of target network update.
            target_update_freq=100,
        ),
        # collect_mode config
        collect=dict(
            # You can use either "n_sample" or "n_episode" in collector.collect.
            # Get "n_sample" samples per collect.
            n_sample=64,
            # Cut trajectories into pieces with length "unroll_len".
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
                decay=50000,
            ),
            replay_buffer=dict(replay_buffer_size=100000, )
        ),
    ),
)
lunarlander_dqn_gail_default_config = EasyDict(lunarlander_dqn_gail_default_config)
main_config = lunarlander_dqn_gail_default_config

lunarlander_dqn_gail_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['dizoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='dqn'),
)
lunarlander_dqn_gail_create_config = EasyDict(lunarlander_dqn_gail_create_config)
create_config = lunarlander_dqn_gail_create_config

if __name__ == "__main__":
    serial_pipeline_gail(
        [main_config, create_config], [lunarlander_dqn_default_config, lunarlander_dqn_create_config],
        seed=0,
        collect_data=True
    )
