from easydict import EasyDict
from ding.entry import serial_pipeline_guided_cost

lunarlander_ppo_config = dict(
    exp_name='lunarlander_guided_cost',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        env_id='LunarLander-v2',
        n_evaluator_episode=5,
        stop_value=200,
    ),
    reward_model=dict(
        learning_rate=0.001,
        input_size=9,
        batch_size=32,
        continuous=False,
        update_per_collect=20,
    ),
    policy=dict(
        cuda=False,
        action_space='discrete',
        recompute_adv=True,
        model=dict(
            obs_shape=8,
            action_shape=4,
            action_space='discrete',
        ),
        learn=dict(
            update_per_collect=8,
            batch_size=800,
            learning_rate=0.001,
            value_weight=0.5,
            entropy_weight=0.01,
            clip_ratio=0.2,
            adv_norm=True,
        ),
        collect=dict(
            demonstration_info_path='path',
            n_sample=800,
            unroll_len=1,
            discount_factor=0.99,
            gae_lambda=0.95,
        ),
    ),
)
lunarlander_ppo_config = EasyDict(lunarlander_ppo_config)
main_config = lunarlander_ppo_config
lunarlander_ppo_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['dizoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='ppo'),
    reward_model=dict(type='guided_cost'),
)
lunarlander_ppo_create_config = EasyDict(lunarlander_ppo_create_config)
create_config = lunarlander_ppo_create_config

if __name__ == "__main__":
    serial_pipeline_guided_cost([main_config, create_config], seed=0)
