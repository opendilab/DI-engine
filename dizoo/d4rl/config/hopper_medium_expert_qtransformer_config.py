# You can conduct Experiments on D4RL with this config file through the following command:
# cd ../entry && python d4rl_qtransformer_main.py
from easydict import EasyDict

main_config = dict(
    exp_name="hopper_medium_expert_qtransformer_seed0",
    env=dict(
        env_id='hopper-medium-expert-v0',
        collector_env_num=5,
        evaluator_env_num=8,
        use_act_scale=True,
        n_evaluator_episode=8,
        stop_value=6000,
    ),

    policy=dict(
        cuda=True,
        
        model=dict(
            num_actions = 3,
            action_bins = 16,
            obs_dim = 11,
            dueling = False,
            attend_dim = 512,
        ),
        
        learn=dict(
            data_path=None,
            train_epoch=3000,
            batch_size=2048,
            learning_rate_q=3e-4,
            alpha=0.2,
            discount_factor_gamma=0.99,
            min_reward = 0.0,
            auto_alpha=False,
            lagrange_thresh=-1.0,
            min_q_weight=5.0,
        ),
        collect=dict(data_type='d4rl', ),
        eval=dict(evaluator=dict(eval_freq=5, )),
        other=dict(replay_buffer=dict(replay_buffer_size=2000000, ),
                    low = [-1, -1, -1],
                    high = [1, 1, 1],
         ),
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
        type='qtransformer',
        import_names=['ding.policy.qtransformer'],
    ),
    replay_buffer=dict(type='naive', ),
)
create_config = EasyDict(create_config)
create_config = create_config
