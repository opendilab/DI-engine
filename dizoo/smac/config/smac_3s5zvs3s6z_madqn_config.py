from ding.entry import serial_pipeline
from easydict import EasyDict

agent_num = 8
collector_env_num = 4
evaluator_env_num = 8

main_config = dict(
    exp_name='3s5z3s6z_global_eval_seed1_10w_unroll10_b32_u40',
    env=dict(
        map_name='3s5z_vs_3s6z',
        difficulty=7,
        reward_only_positive=True,
        mirror_opponent=False,
        agent_num=agent_num,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        stop_value=1.999,
        n_evaluator_episode=32,
        special_global_state=True,
        manager=dict(
            shared_memory=False,
        ),
    ),
    policy=dict(
        nstep=3,
        model=dict(
            agent_num=agent_num,
            obs_shape=159,
            global_obs_shape=314,
            global_boost=True,
            action_shape=15,
            hidden_size_list=[256, 256],
            mixer=False,
            lstm_type='gru',
            dueling=False,
        ),
        learn=dict(
            multi_gpu=False,
            update_per_collect=40,
            batch_size=32,
            learning_rate=0.0005,
            clip_value=5,
            double_q=False,
            iql=False,
            target_update_theta=0.008,
            discount_factor=0.95,
        ),
        collect=dict(
            n_episode=32,
            unroll_len=10,
            env_num=collector_env_num,
        ),
        eval=dict(env_num=evaluator_env_num, evaluator=dict(eval_freq=500, )),
        other=dict(
            eps=dict(
                type='linear',
                start=1,
                end=0.05,
                decay=100000,
            ),
            replay_buffer=dict(
                replay_buffer_size=30000,
                # (int) The maximum reuse times of each data
                max_reuse=1e+9,
                max_staleness=1e+9,
            ),
        ),
    ),
)
main_config = EasyDict(main_config)
create_config = dict(
    env=dict(
        type='smac',
        import_names=['dizoo.smac.envs.smac_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='madqn'),
    collector=dict(type='episode', get_train_sample=True),
)
create_config = EasyDict(create_config)


def train(args):
    config = [main_config, create_config]
    serial_pipeline(config, seed=args.seed)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=0)
    args = parser.parse_args()

    train(args)
