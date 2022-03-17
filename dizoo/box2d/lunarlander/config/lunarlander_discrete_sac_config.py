from easydict import EasyDict
from ding.entry import serial_pipeline

lunarlander_sac_default_config = dict(
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        env_id='LunarLander-v2',
        n_evaluator_episode=5,
        stop_value=195,
    ),
    policy=dict(
        cuda=False,
        random_collect_size=0,
        multi_agent=False,
        model=dict(
            agent_obs_shape=8,
            global_obs_shape=8,
            action_shape=4,
            twin_critic=True,
            actor_head_hidden_size=64,
            critic_head_hidden_size=64,
        ),
        learn=dict(
            update_per_collect=2,
            batch_size=64,
            learning_rate_q=5e-3,
            learning_rate_policy=5e-3,
            learning_rate_alpha=3e-4,
            ignore_done=False,
            target_theta=0.01,
            discount_factor=0.99,
            alpha=0.2,
            auto_alpha=False,
        ),
        collect=dict(
            env_num=8,
            n_sample=256,
            unroll_len=1,
        ),
        command=dict(),
        eval=dict(
            evaluator=dict(eval_freq=50, ),
            env_num=5,
        ),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.1,
                decay=50000,
            ), replay_buffer=dict(replay_buffer_size=100000, )
        ),
    ),
)

lunarlander_sac_default_config = EasyDict(lunarlander_sac_default_config)
main_config = lunarlander_sac_default_config

lunarlander_sac_default_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['dizoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='sac_discrete', ),
)
lunarlander_sac_default_create_config = EasyDict(lunarlander_sac_default_create_config)
create_config = lunarlander_sac_default_create_config

if __name__ == "__main__":
    serial_pipeline([main_config, create_config], seed=0)
