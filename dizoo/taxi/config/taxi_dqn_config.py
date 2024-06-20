from easydict import EasyDict

taxi_dqn_config = dict(
    exp_name='taxi_dqn_seed0',
    env=dict(
        collector_env_num=8,
        evaluator_env_num=8,
        n_evaluator_episode=8,   
        stop_value=20,           
        max_episode_steps=60,    
        env_id="Taxi-v3" 
    ),
    policy=dict(
        cuda=True,
        model=dict(
            obs_shape=34,
            action_shape=6,
            encoder_hidden_size_list=[128, 128]
        ),
        random_collect_size=5000,
        nstep=3,
        discount_factor=0.99,
        learn=dict(
            update_per_collect=10,
            batch_size=64,
            learning_rate=0.0001,
            learner=dict(
                hook=dict(
                    log_show_after_iter=1000,
                )
            ),
        ),
        collect=dict(n_sample=32),
        eval=dict(evaluator=dict(eval_freq=1000, )), 
        other=dict(
            eps=dict(
              type="linear",
              start=1,
              end=0.05,
              decay=3000000                             
            ),                                      
            replay_buffer=dict(replay_buffer_size=100000,),  
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
    serial_pipeline((main_config, create_config), max_env_step=3000000, seed=0)