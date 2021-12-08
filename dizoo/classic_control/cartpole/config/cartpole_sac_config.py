from easydict import EasyDict
from ding.entry import serial_pipeline

cartpole_sac_config = dict(
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=195,
    ),
    policy=dict(
        cuda=False,
        random_collect_size=0,
        multi_agent=False,
        model=dict(
            agent_obs_shape=4,
            global_obs_shape=4,
            action_shape=2,
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

cartpole_sac_config = EasyDict(cartpole_sac_config)
main_config = cartpole_sac_config

cartpole_sac_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='sac_discrete', ),
)
cartpole_sac_create_config = EasyDict(cartpole_sac_create_config)
create_config = cartpole_sac_create_config

if __name__ == "__main__":
    serial_pipeline([main_config, create_config], seed=0)
