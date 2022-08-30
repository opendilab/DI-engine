from copy import deepcopy
from easydict import EasyDict

spaceinvaders_a2c_config = dict(
    exp_name='spaceinvaders_a2c_seed0',
    env=dict(
        collector_env_num=16,
        evaluator_env_num=8,
        n_evaluator_episode=8,
        stop_value=10000000000,
        env_id='SpaceInvaders-v4',
        #'ALE/SpaceInvaders-v5' is available. But special setting is needed after gym make.
        frame_stack=4,
    ),
    policy=dict(
        cuda=True,
        model=dict(
            obs_shape=[4, 84, 84],
            action_shape=6,
            encoder_hidden_size_list=[32, 64, 64, 128],
            actor_head_hidden_size=128,
            critic_head_hidden_size=128,
            critic_head_layer_num=2,
        ),
        learn=dict(
            batch_size=80,
            # (bool) Whether to normalize advantage. Default to False.
            adv_norm=False,
            learning_rate=0.0001,
            # (float) loss weight of the value network, the weight of policy network is set to 1
            value_weight=0.5,
            # (float) loss weight of the entropy regularization, the weight of policy network is set to 1
            entropy_weight=0.01,
            grad_norm=0.5,
            betas=(0.3, 0.99),
        ),
        collect=dict(
            # (int) collect n_sample data, train model 1 times
            n_sample=80,
            # (float) the trade-off factor lambda to balance 1step td and mc
            gae_lambda=0.99,
            discount_factor=0.99,
        ),
        eval=dict(evaluator=dict(eval_freq=500, )),
    ),
)
spaceinvaders_a2c_config = EasyDict(spaceinvaders_a2c_config)
main_config = spaceinvaders_a2c_config

spaceinvaders_a2c_create_config = dict(
    env=dict(
        type='atari',
        import_names=['dizoo.atari.envs.atari_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='a2c'),
)
spaceinvaders_a2c_create_config = EasyDict(spaceinvaders_a2c_create_config)
create_config = spaceinvaders_a2c_create_config

if __name__ == '__main__':
    # or you can enter ding -m serial_onpolicy -c spaceinvaders_a2c_config.py -s 0
    from ding.entry import serial_pipeline
    serial_pipeline((main_config, create_config), seed=0)
