from easydict import EasyDict
from ding.entry import serial_pipeline

ant_sac_default_config = dict(
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=195,
    ),
    policy=dict(
        cuda=False,
        on_policy=False,
        random_collect_size=10000,
        model=dict(
            agent_obs_shape=4,
            global_obs_shape=4,
            action_shape=2,
            twin_critic=True,
            actor_head_hidden_size=256,
            critic_head_hidden_size=256,
        ),
        learn=dict(
            update_per_collect=1,
            batch_size=256,
            learning_rate_q=1e-3,
            learning_rate_policy=1e-3,
            learning_rate_alpha=3e-4,
            ignore_done=False,
            target_theta=0.005,
            discount_factor=0.99,
            alpha=0.2,
            auto_alpha=False,
        ),
        collect=dict(
            n_sample=1,
            unroll_len=1,
        ),
        command=dict(),
        eval=dict(),
        other=dict(replay_buffer=dict(replay_buffer_size=1000000, ), ),
    ),
)

ant_sac_default_config = EasyDict(ant_sac_default_config)
main_config = ant_sac_default_config

ant_sac_default_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(
        type='masac',
    ),
    replay_buffer=dict(type='naive', ),
)
ant_sac_default_create_config = EasyDict(ant_sac_default_create_config)
create_config = ant_sac_default_create_config


if __name__ == "__main__":
    serial_pipeline([main_config, create_config], seed=0)