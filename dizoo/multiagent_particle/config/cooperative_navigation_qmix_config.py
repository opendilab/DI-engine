from copy import deepcopy
from ding.entry import serial_pipeline
from easydict import EasyDict

n_agent = 5
collector_env_num = 4
evaluator_env_num = 2
num_landmarks = n_agent
main_config = dict(
    env=dict(
        num_landmarks=num_landmarks,
        max_step=100,
        n_agent=n_agent,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        manager=dict(shared_memory=False, ),
        n_evaluator_episode=5,
        stop_value=0,
    ),
    policy=dict(
        model=dict(
            agent_num=n_agent,
            obs_shape=2 + 2 + (n_agent - 1) * 2 + num_landmarks * 2,
            global_obs_shape=n_agent * 2 + num_landmarks * 2 + n_agent * 2,
            action_shape=5,
            hidden_size_list=[128, 128, 64],
            mixer=True,
        ),
        learn=dict(
            update_per_collect=100,
            batch_size=32,
            learning_rate=0.0005,
            target_update_theta=0.001,
            discount_factor=0.99,
            double_q=True,
        ),
        collect=dict(
            n_sample=600,
            unroll_len=16,
            env_num=collector_env_num,
        ),
        eval=dict(env_num=evaluator_env_num, ),
        other=dict(eps=dict(
            type='exp',
            start=1.0,
            end=0.05,
            decay=100000,
        ), ),
    ),
)
main_config = EasyDict(main_config)
create_config = dict(
    env=dict(
        import_names=['dizoo.multiagent_particle.envs.particle_env'],
        type='cooperative_navigation',
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='qmix'),
)
create_config = EasyDict(create_config)

cooperative_navigation_qmix_config = main_config
cooperative_navigation_qmix_create_config = create_config


def train(args):
    config = [main_config, create_config]
    serial_pipeline(config, seed=args.seed)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0)
    args = parser.parse_args()

    train(args)
