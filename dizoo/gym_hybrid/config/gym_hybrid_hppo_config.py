from easydict import EasyDict
from ding.entry import serial_pipeline_onpolicy

gym_hybrid_hppo_config = dict(
    exp_name='gym_hybrid_hppo_seed3',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        # (bool) Scale output action into legal range [-1, 1].
        act_scale=True,
        env_id='Moving-v0',  # ['Sliding-v0', 'Moving-v0']
        n_evaluator_episode=5,
        stop_value=1.8,  # 1.85 for hybrid_hppo
    ),
    policy=dict(
        cuda=True,
        priority=False,
        action_space='hybrid',
        model=dict(
            obs_shape=10,
            action_shape=dict(
                action_type_shape=3,
                action_args_shape=2,
            ),
            action_space='hybrid',
        ),
        learn=dict(
            epoch_per_collect=1,
            batch_size=40,
            learning_rate=0.0001,
            entropy_weight=0.00001,
            adv_norm=False,
        ),
        collect=dict(
            n_sample=120,
            discount_factor=0.95,
            gae_lambda=0.97,
            collector=dict(collect_print_freq=1000, ),
        ),
        eval=dict(evaluator=dict(eval_freq=200, ), ),
    ),
)
gym_hybrid_hppo_config = EasyDict(gym_hybrid_hppo_config)
main_config = gym_hybrid_hppo_config

gym_hybrid_hppo_create_config = dict(
    env=dict(
        type='gym_hybrid',
        import_names=['dizoo.gym_hybrid.envs.gym_hybrid_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='ppo'),
)
gym_hybrid_hppo_create_config = EasyDict(gym_hybrid_hppo_create_config)
create_config = gym_hybrid_hppo_create_config

if __name__ == "__main__":
    serial_pipeline_onpolicy([main_config, create_config], seed=0)
