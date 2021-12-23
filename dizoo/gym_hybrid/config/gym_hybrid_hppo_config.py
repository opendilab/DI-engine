from easydict import EasyDict
from ding.entry import serial_pipeline_onpolicy

gym_hybrid_hppo_config = dict(
    exp_name='gym_hybrid_hppo_actsacle_fsv0.3_ew0.001_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        # (bool) Scale output action into legal range, usually [-1, 1].
        act_scale=True,
        env_id='Moving-v0',  # ['Sliding-v0', 'Moving-v0']
        n_evaluator_episode=5,
        stop_value=1.8,  # 1.85 for hybrid_hppo
    ),
    policy=dict(
        cuda=True,
        priority=False,
        action_space='hybrid',
        recompute_adv=True,
        model=dict(
            obs_shape=10,
            action_shape=dict(
                action_type_shape=3,
                action_args_shape=2,
            ),
            action_space='hybrid',
            encoder_hidden_size_list=[256, 128, 64, 64],
            sigma_type='fixed',
            fixed_sigma_value=0.3,  # TODO(pu)
            bound_type='tanh',
        ),
        learn=dict(
            epoch_per_collect=10,
            batch_size=320,
            learning_rate=3e-4,
            value_weight=0.5,
            entropy_weight=0.001,  # TODO(pu)
            clip_ratio=0.2,
            adv_norm=True,
            value_norm=True,
        ),
        collect=dict(
            n_sample=int(3200),
            discount_factor=0.99,
            gae_lambda=0.95,
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
