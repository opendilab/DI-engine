from easydict import EasyDict
from ding.entry import serial_pipeline, serial_pipeline_reward_model

minigrid_dqn_config = dict(
    exp_name="minigrid_empty8_dqn",
    env=dict(
        collector_env_num=8,
        evaluator_env_num=5,
        env_id='MiniGrid-Empty-8x8-v0',
        n_evaluator_episode=5,
        stop_value=0.96,
    ),
    policy=dict(
        cuda=True,
        priority=False,
        model=dict(
            obs_shape=2739,
            action_shape=7,
            encoder_hidden_size_list=[256, 128, 64, 64],
        ),
        nstep=3,
        discount_factor=0.99,
        learn=dict(
            update_per_collect=10,
            batch_size=32,
            learning_rate=0.0001,
            target_update_freq=500,
        ),
        collect=dict(n_sample=96, ),
        eval=dict(evaluator=dict(eval_freq=1000, )),
        other=dict(
            eps=dict(
                type='exp',
                start=0.01,
                end=0.01,
                decay=250000,
            ),
            replay_buffer=dict(replay_buffer_size=500000, ),
        ),
    ),
    reward_model=dict(
        type='countbased',
        counter_type='SimHash',
        bonus_coefficent=0.1,
        state_dim=2739,
        hash_dim=64,
    )
)
minigrid_dqn_config = EasyDict(minigrid_dqn_config)
main_config = minigrid_dqn_config
minigrid_dqn_create_config = dict(
    env=dict(
        type='minigrid',
        import_names=['dizoo.minigrid.envs.minigrid_env'],
    ),
    env_manager=dict(type='base'),
    policy=dict(type='dqn'),
)
minigrid_dqn_create_config = EasyDict(minigrid_dqn_create_config)
create_config = minigrid_dqn_create_config

if __name__ == "__main__":
    serial_pipeline_reward_model([main_config, create_config], seed=0)
