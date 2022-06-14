from easydict import EasyDict
from cartpole_dqn_config import cartpole_dqn_config, cartpole_dqn_create_config

cartpole_expert_model_config = dict(
    exp_name='cartpole_bco_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=195,
        replay_path='cartpole_dqn/video',
    ),
    policy=dict(
        cuda=True,
        model=dict(
            obs_shape=4,
            action_shape=2,
            encoder_hidden_size_list=[128, 128, 64],
            dueling=True,
        ),
        learn=dict(
            multi_gpu=False,
            bp_update_sync=False,
            train_epoch=200,  # If train_epoch is 1, the algorithm will be BCO(0)
            batch_size=32,
            learning_rate=0.01,
            decay_epoch=30,
            decay_rate=0.1,
            warmup_lr=1e-4,
            warmup_epoch=3,
            weight_decay=1e-4,
        ),
        collect=dict(
            n_episode=10,
            # control the number (alpha*n_episode) of post-demonstration environment interactions at each iteration.
            # Notice: alpha * n_episode > collector_env_num
            alpha=0.8,
            demonstration_model_path='abs model path',
            demonstration_offline_data_path='abs data path',
        ),
        eval=dict(evaluator=dict(eval_freq=40, )),
        other=dict(eps=dict(
            type='exp',
            start=0.95,
            end=0.1,
            decay=10000,
        ), )
    ),
)
cartpole_expert_model_config = EasyDict(cartpole_expert_model_config)
main_config = cartpole_expert_model_config
cartpole_expert_model_create_config = dict(
    env=dict(
        type='cartpole',
        import_names=['dizoo.classic_control.cartpole.envs.cartpole_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='bco'),
    collector=dict(type='episode')
)
cartpole_expert_model_create_config = EasyDict(cartpole_expert_model_create_config)
create_config = cartpole_expert_model_create_config

if __name__ == "__main__":
    from ding.entry import serial_pipeline_bco
    serial_pipeline_bco(
        [main_config, create_config], [cartpole_dqn_config, cartpole_dqn_create_config], seed=0, max_env_step=100000
    )
