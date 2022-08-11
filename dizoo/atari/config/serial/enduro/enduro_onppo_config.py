from easydict import EasyDict

enduro_onppo_config = dict(
    exp_name='enduro_onppo_seed0',
    env=dict(
        collector_env_num=64,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=10000000000,
        env_id='Enduro-v4',
        #'ALE/Enduro-v5' is available. But special setting is needed after gym make.
        frame_stack=4,
        manager=dict(shared_memory=False, )
    ),
    policy=dict(
        cuda=True,
        model=dict(
            obs_shape=[4, 84, 84],
            action_shape=9,
            encoder_hidden_size_list=[32, 64, 64, 128],
            actor_head_hidden_size=128,
            critic_head_hidden_size=128,
            critic_head_layer_num=2,
        ),
        learn=dict(
            update_per_collect=24,
            batch_size=128,
            # (bool) Whether to normalize advantage. Default to False.
            adv_norm=False,
            learning_rate=0.0001,
            # (float) loss weight of the value network, the weight of policy network is set to 1
            value_weight=1.0,
            # (float) loss weight of the entropy regularization, the weight of policy network is set to 1
            entropy_weight=0.001,  # [0.1, 0.01 ,0.0]
            clip_ratio=0.1
        ),
        collect=dict(
            # (int) collect n_sample data, train model n_iteration times
            n_sample=1024,
            # (float) the trade-off factor lambda to balance 1step td and mc
            gae_lambda=0.95,
            discount_factor=0.99,
        ),
        eval=dict(evaluator=dict(eval_freq=1000, )),
        other=dict(replay_buffer=dict(
            replay_buffer_size=10000,
            max_use=3,
        ), ),
    ),
)
main_config = EasyDict(enduro_onppo_config)

enduro_onppo_create_config = dict(
    env=dict(
        type='atari',
        import_names=['dizoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='ppo'),
)
create_config = EasyDict(enduro_onppo_create_config)

if __name__ == '__main__':
    # or you can enter ding -m serial_onpolicy -c enduro_onppo_config.py -s 0
    from ding.entry import serial_pipeline_onpolicy
    serial_pipeline_onpolicy((main_config, create_config), seed=0)
