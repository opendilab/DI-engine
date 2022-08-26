from easydict import EasyDict

collector_env_num = 8
evaluator_env_num = 5

gfootball_dqn_main_config = dict(
    exp_name='gfootball_dqn_seed0',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        stop_value=999,
        env_name="11_vs_11_easy_stochastic",
        # env_name="11_vs_11_stochastic",  # default: medium
        # env_name="11_vs_11_hard_stochastic",
        save_replay_gif=False,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        cuda=True,
        nstep=5,
        discount_factor=0.997,
        model=dict(),
        learn=dict(
            update_per_collect=20,
            batch_size=512,
            learning_rate=0.0001,
            target_update_freq=500,
        ),
        collect=dict(n_sample=256),
        eval=dict(evaluator=dict(eval_freq=5000)),
        other=dict(
            eps=dict(
                type='exp',
                start=1,
                end=0.05,
                decay=int(2e5),
            ),
            replay_buffer=dict(replay_buffer_size=int(5e5), ),
        ),
    ),
)
gfootball_dqn_main_config = EasyDict(gfootball_dqn_main_config)
main_config = gfootball_dqn_main_config

gfootball_dqn_create_config = dict(
    env=dict(
        type='gfootball',
        import_names=['dizoo.gfootball.envs.gfootball_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='dqn'),
)
gfootball_dqn_create_config = EasyDict(gfootball_dqn_create_config)
create_config = gfootball_dqn_create_config

if __name__ == '__main__':
    from ding.entry import serial_pipeline
    from dizoo.gfootball.model.q_network.football_q_network import FootballNaiveQ
    football_naive_q = FootballNaiveQ()
    serial_pipeline((main_config, create_config), model=football_naive_q, seed=0, max_env_step=int(5e6))
