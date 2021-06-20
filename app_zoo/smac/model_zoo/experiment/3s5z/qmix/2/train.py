from copy import deepcopy
from nervex.entry import serial_pipeline
from easydict import EasyDict

agent_num = 8
collector_env_num = 16
evaluator_env_num = 8

main_config = dict(
    env=dict(
        map_name='3s5z',
        difficulty=7,
        reward_only_positive=True,
        mirror_opponent=False,
        agent_num=agent_num,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        shared_memory=False,
        stop_value=0.999,
        n_evaluator_episode=32,
    ),
    policy=dict(
        model=dict(
            # (int) agent_num: The number of the agent.
            # For SMAC 3s5z, agent_num=8; for 2c_vs_64zg, agent_num=2.
            agent_num=agent_num,
            # (int) obs_shape: The shapeension of observation of each agent.
            # For 3s5z, obs_shape=150; for 2c_vs_64zg, agent_num=404.
            obs_shape=150,
            # (int) global_obs_shape: The shapeension of global observation.
            # For 3s5z, obs_shape=216; for 2c_vs_64zg, agent_num=342.
            global_obs_shape=216,
            # (int) action_shape: The number of action which each agent can take.
            # action_shape= the number of common action (6) + the number of enemies.
            # For 3s5z, obs_shape=14 (6+8); for 2c_vs_64zg, agent_num=70 (6+64).
            action_shape=14,
            # (List[int]) The size of hidden layer
            hidden_size_list=[32],
            # (bool) Whether to use mixer for q_value aggregation
            mixer=True,
            use_gru=True,
            use_pmixer=True,
        ),
        # used in state_num of hidden_state
        collect=dict(
            unroll_len=30,
            env_num=collector_env_num,
        ),
        eval=dict(env_num=evaluator_env_num),
        other=dict(
            replay_buffer=dict(
                replay_buffer_size=15000,
            ),
        ),
    ),
)
main_config = EasyDict(main_config)
create_config = dict(
    env=dict(
        type='smac',
        import_names=['app_zoo.smac.envs.smac_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='qmix'),
)
create_config = EasyDict(create_config)

def train_dqn(args):
    config = [main_config, create_config]
    serial_pipeline(config, seed=args.seed)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0)
    args = parser.parse_args()

    train_dqn(args)
