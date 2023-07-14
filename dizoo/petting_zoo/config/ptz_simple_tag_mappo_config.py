from easydict import EasyDict

num_good=1
num_adversaries=3
num_obstacles=2
collector_env_num = 8
evaluator_env_num = 8
agent_obs_shape = [2+2+2*num_obstacles+2*(num_good+num_adversaries-1)+2*(num_good-1), 2+2+2*num_obstacles+2*(num_good+num_adversaries-1)+2*num_good]
global_obs_shape=[agent_obs_shape[0]+(2+2)*(num_good+num_adversaries)+2*num_obstacles,agent_obs_shape[1]+(2+2)*(num_good+num_adversaries)+2*num_obstacles]
            
main_config = dict(
    exp_name='ptz_simple_tag_mappo_seed0',
    env=dict(
        env_family='mpe',
        env_id='simple_tag_v2',
        num_good=num_good,
        num_adversaries=num_adversaries,
        num_obstacles=num_obstacles,
        max_cycles=25,
        agent_obs_only=False,
        agent_specific_global_state=True,
        continuous_actions=False,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        # stop_value=0,
    ),
    policy=dict(
        cuda=True,
        multi_agent=True,
        dict_obs = True,
        action_space='discrete',
        model=dict(
            action_space='discrete',
            agent_obs_shape=agent_obs_shape,
            global_obs_shape=global_obs_shape,
            action_shape=5,
            num_good=num_good,
            num_adversaries=num_adversaries,
            num_obstacles=num_obstacles,
            # actor_encoder_hidden_size=256,
            # critic_encoder_hidden_size = 256,
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
        eval=dict(
            env_num=evaluator_env_num,
            evaluator=dict(eval_freq=50, ),
        ),
        other=dict(),
    ),
)
main_config = EasyDict(main_config)
create_config = dict(
    env=dict(
        import_names=['dizoo.petting_zoo.envs.petting_zoo_simple_tag_env'],
        type='petting_zoo_tag',
    ),
    env_manager=dict(type='base'),
    policy=dict(type='ppo'),
)
create_config = EasyDict(create_config)
ptz_simple_tag_mappo_config = main_config
ptz_simple_tag_mappo_create_config = create_config

if __name__ == '__main__':
    # or you can enter `ding -m serial_onpolicy -c ptz_simple_spread_mappo_config.py -s 0`
    from ding.entry import serial_pipeline_onpolicy
    serial_pipeline_onpolicy((main_config, create_config), seed=0)
