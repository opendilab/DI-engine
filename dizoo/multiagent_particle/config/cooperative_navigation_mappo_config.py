from copy import deepcopy
from ding.entry import serial_pipeline_onpolicy
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
        cuda=False,
        multi_agent=True,
        continuous=False,
        model=dict(
            agent_num=n_agent,
            agent_obs_shape=2 + 2 + (n_agent - 1) * 2 + num_landmarks * 2,
            global_obs_shape=n_agent * 2 + num_landmarks * 2 + n_agent * 2,
            action_shape=5,
        ),
        learn=dict(
            multi_gpu=False,
            epoch_per_collect=5,
            batch_size=3200,
            learning_rate=5e-4,
            # ==============================================================
            # The following configs is algorithm-specific
            # ==============================================================
            # (float) The loss weight of value network, policy network weight is set to 1
            value_weight=0.5,
            # (float) The loss weight of entropy regularization, policy network weight is set to 1
            entropy_weight=0.01,
            # (float) PPO clip ratio, defaults to 0.2
            clip_ratio=0.2,
            # (bool) Whether to use advantage norm in a whole training batch
            adv_norm=False,
            value_norm=True,
            ppo_param_init=True,
            grad_clip_type='clip_norm',
            grad_clip_value=10,
            ignore_done=False,
        ),
        collect=dict(
            n_sample=3200,
            unroll_len=1,
            env_num=collector_env_num,
        ),
        on_policy=True,
        eval=dict(
            env_num=evaluator_env_num,
            evaluator=dict(eval_freq=50, ),
        ),
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
    env_manager=dict(type='base'),
    policy=dict(type='ppo'),
)
create_config = EasyDict(create_config)

cooperative_navigation_mappo_config = main_config
cooperative_navigation_mappo_create_config = create_config


def train(args):
    config = [main_config, create_config]
    serial_pipeline_onpolicy(config, seed=args.seed)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0)
    args = parser.parse_args()

    train(args)
