# You can conduct Experiments on D4RL with this config file through the following command:
# cd ../entry && python d4rl_iql_main.py
from easydict import EasyDict

main_config = dict(
    exp_name="halfcheetah_medium_expert_iql_seed0",
    env=dict(
        env_id='halfcheetah-medium-expert-v2',
        collector_env_num=1,
        evaluator_env_num=8,
        use_act_scale=True,
        n_evaluator_episode=8,
        stop_value=6000,
        reward_norm="iql_locomotion",
    ),
    policy=dict(
        cuda=True,
        model=dict(
            obs_shape=17,
            action_shape=6,
        ),
        learn=dict(
            data_path=None,
            train_epoch=30000,
            batch_size=4096,
            learning_rate_q=3e-4,
            learning_rate_policy=1e-4,
            beta=0.05,
            tau=0.7,
        ),
        collect=dict(data_type='d4rl', ),
        eval=dict(evaluator=dict(eval_freq=5000, )),
        other=dict(replay_buffer=dict(replay_buffer_size=2000000, ), ),
    ),
)

main_config = EasyDict(main_config)
main_config = main_config

create_config = dict(
    env=dict(
        type='d4rl',
        import_names=['dizoo.d4rl.envs.d4rl_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(
        type='iql',
        import_names=['ding.policy.iql'],
    ),
    replay_buffer=dict(type='naive', ),
)
create_config = EasyDict(create_config)
create_config = create_config
