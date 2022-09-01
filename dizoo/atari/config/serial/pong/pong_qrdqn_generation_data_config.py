from easydict import EasyDict

pong_qrdqn_config = dict(
    exp_name='pong_qrdqn_generation_data_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=20,
        env_id='Pong-v4',
        #'ALE/Pong-v5' is available. But special setting is needed after gym make.
        frame_stack=4,
    ),
    policy=dict(
        cuda=True,
        priority=False,
        model=dict(
            obs_shape=[4, 84, 84],
            action_shape=6,
            encoder_hidden_size_list=[128, 128, 512],
            num_quantiles=64,
        ),
        nstep=1,
        discount_factor=0.99,
        collect=dict(
            collect_count=1000,
            data_type='hdf5',
            # pretrained RL model path, user can modify it as its own path
            model_path='./pong_qrdqn_seed0/ckpt/ckpt_best.pth.tar',
            # this prefix should be the same as exp_name
            expert_data_path='./pong_qrdqn_generation_data_seed0/expert.pkl',
        ),
        eval=dict(evaluator=dict(eval_freq=4000, )),
        other=dict(
            eps=dict(
                type='exp',
                start=1.,
                end=0.05,
                decay=250000,
                collect=0.2,
            ),
            replay_buffer=dict(replay_buffer_size=100000, ),
        ),
    ),
)
pong_qrdqn_config = EasyDict(pong_qrdqn_config)
main_config = pong_qrdqn_config
pong_qrdqn_create_config = dict(
    env=dict(
        type='atari',
        import_names=['dizoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='qrdqn'),
)
pong_qrdqn_create_config = EasyDict(pong_qrdqn_create_config)
create_config = pong_qrdqn_create_config

if __name__ == "__main__":
    from ding.entry import collect_demo_data
    cfg = main_config.policy.collect
    collect_demo_data(
        (main_config, create_config),
        seed=0,
        collect_count=cfg.collect_count,
        expert_data_path=cfg.expert_data_path,
        state_dict_path=cfg.model_path
    )
