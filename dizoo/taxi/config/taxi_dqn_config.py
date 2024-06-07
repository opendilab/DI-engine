from easydict import EasyDict

taxi_dqn_config = dict(
    exp_name='taxi_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=8,  
        n_evaluator_episode=10,
        stop_value=500_000_000,
        max_episode_steps=100,
        env_id="Taxi-v3"
    ),
    policy=dict(
        cuda=True,
        load_path="./taxi_dqn_seed0/ckpt/ckpt_best.pth.tar",
        model=dict(
            obs_shape=34,    
            action_shape=6,
            encoder_hidden_size_list=[128, 128]
        ),
        nstep=3,
        discount_factor=0.99,
        learn=dict(
            update_per_collect=5,
            batch_size=64,
            learning_rate=0.0001,
        ),
        collect=dict(n_sample=32),
        eval=dict(evaluator=dict(eval_freq=5000, )),
        other=dict(
            eps=dict(
              type="linear",
              start=1,
              end=0.01,
              decay=1000000  
            ),
            replay_buffer=dict(replay_buffer_size=10000,),
        ),
    )
)
taxi_dqn_config = EasyDict(taxi_dqn_config)
main_config = taxi_dqn_config

taxi_dqn_create_config = dict(
    env=dict(
        type="taxi",
        import_names=["dizoo.taxi.envs.taxi_env"]
    ),
    env_manager=dict(type='base'),
    policy=dict(type='dqn'),
    replay_buffer=dict(type='deque', import_names=['ding.data.buffer.deque_buffer_wrapper']),
)

taxi_dqn_create_config = EasyDict(taxi_dqn_create_config)
create_config = taxi_dqn_create_config

if __name__ == "__main__":
    from ding.entry import serial_pipeline
    serial_pipeline((main_config, create_config), max_env_step=1000000, seed=0)
