from easydict import EasyDict

collector_env_num = 8
evaluator_env_num = 8
n_agent = 2

main_config = dict(
    exp_name='HAPPO_result/debug/multi_mujoco_walker_2x3_happo',
    env=dict(
        scenario='Walker2d-v2',
        agent_conf="2x3",
        agent_obsk=2,
        add_agent_id=False,
        episode_limit=1000,
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=8,
        stop_value=6000,
    ),
    policy=dict(
        cuda=True,
        multi_agent=True,
        agent_num = n_agent,
        action_space='continuous',
        model=dict(
            action_space='continuous',
            agent_num=n_agent,
            agent_obs_shape=8,
            global_obs_shape=17,
            action_shape=3,
            use_lstm=False,
        ),
        learn=dict(
            multi_gpu=False,
            epoch_per_collect=5,
            # batch_size=3200,
            # batch_size=800,
            batch_size=320,
            # learning_rate=5e-4,
            learning_rate=3e-3,
            # ==============================================================
            # The following configs is algorithm-specific
            # ==============================================================
            # (float) The loss weight of value network, policy network weight is set to 1
            value_weight=0.5,
            # value_weight=1,
            # (float) The loss weight of entropy regularization, policy network weight is set to 1
            entropy_weight=0.001,
            # entropy_weight=0.01,
            # (float) PPO clip ratio, defaults to 0.2
            clip_ratio=0.2,
            # (bool) Whether to use advantage norm in a whole training batch
            adv_norm=True,
            value_norm=True,
            ppo_param_init=True,
            grad_clip_type='clip_norm',
            grad_clip_value=5,
            # ignore_done=True,
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
        type='mujoco_multi',
        import_names=['dizoo.multiagent_mujoco.envs.multi_mujoco_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='happo'),
)
create_config = EasyDict(create_config)

# if __name__ == '__main__':
#     from ding.entry import serial_pipeline_onpolicy
#     main_config.exp_name='HAPPO_result/walker/multi_mujoco_walker_2x3_happo'
#     serial_pipeline_onpolicy((main_config, create_config), seed=0, max_env_step=int(1e7))
from ding.entry import serial_pipeline_onpolicy
def train(mconfig, cconfig, seed):
    mconfig.exp_name='HAPPO_result/walker/multi_mujoco_walker_2x3_happo_bs320_lr3e3_idf_ew001_seed{}'.format(seed)
    serial_pipeline_onpolicy((mconfig, cconfig), seed=seed, max_env_step=int(1e7))


if __name__ == '__main__':
    for seed in [0, 1, 2]:
        mconfig = main_config
        cconfig = create_config
        train(mconfig, cconfig, seed)