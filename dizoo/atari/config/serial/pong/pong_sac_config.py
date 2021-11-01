from easydict import EasyDict
from ding.entry import serial_pipeline

pong_sac_default_config = dict(
    env=dict(
        collector_env_num=16,
        evaluator_env_num=4,
        n_evaluator_episode=8,
        stop_value=20,
        env_id='PongNoFrameskip-v4',
        frame_stack=4,
        manager=dict(shared_memory=False, )
    ),
    policy=dict(
        cuda=True,
        on_policy=False,
        multi_agent = False,
        random_collect_size=0,
        model=dict(
            agent_obs_shape=[4, 84, 84],
            global_obs_shape=[4, 84, 84],
            action_shape=6,
            twin_critic=True,
            encoder_hidden_size_list=[64, 64, 128, 256],
            actor_head_hidden_size=256,
            critic_head_hidden_size=256,
        ),
        learn=dict(
            update_per_collect=50,
            batch_size=320,
            learning_rate_q=5e-4,
            learning_rate_policy=5e-4,
            learning_rate_alpha=3e-4,
            ignore_done=False,
            target_theta=0.005,
            discount_factor=0.99,
            alpha=0.2,
            auto_alpha=True,
            log_space=True,
        ),
        collect=dict(
            env_num=16,
            n_sample=3200,
            unroll_len=1,
        ),
        command=dict(),
        eval=dict(
            evaluator=dict(
                eval_freq=50,
            ),
            env_num=4,
        ),
        other=dict(
            eps=dict(
                type='linear',
                start=1,
                end=0.05,
                decay=100000,
            ),  # TODO(pu)
            replay_buffer=dict(replay_buffer_size=200000, ), 
        ),
    ),
)

pong_sac_default_config = EasyDict(pong_sac_default_config)
main_config = pong_sac_default_config

pong_sac_default_create_config = dict(
    env=dict(
        type='atari',
        import_names=['dizoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(
        type='sac_discrete',
    ),
    #replay_buffer=dict(type='naive', ),
)
pong_sac_default_create_config = EasyDict(pong_sac_default_create_config)
create_config = pong_sac_default_create_config


if __name__ == "__main__":
    serial_pipeline([main_config, create_config], seed=0)