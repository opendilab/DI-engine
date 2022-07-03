from easydict import EasyDict

collector_env_num = 8
evaluator_env_num = 5

gfootball_dqn_main_config = dict(
    exp_name='data_gfootball/gfootball_easy_dqn_seed0_n5_df0997_rbs5e5_ed2e5',
    # exp_name='data_gfootball/gfootball_easy_dqn_pretrain_seed0_rbs5e5_ed2e5',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        stop_value=999,
        # env_name="11_vs_11_hard_stochastic",
        # env_name="11_vs_11_stochastic",  # default: medium
        env_name="11_vs_11_easy_stochastic",
        # save_replay_gif=True,
        save_replay_gif=False,
    ),
    policy=dict(
        il_model_path=None,
        rl_model_path=None,
        replay_path=None,

        # il pretrain model
        # il_model_path='/mnt/lustre/puyuan/DI-engine/data_gfootball/gfootball_easy_il_rule_200ep_lt0_seed2/ckpt/ckpt_best.pth.tar',
        # il_model_path='/home/puyuan/DI-engine/data_gfootball/gfootball_easy_il_rule_200ep_lt0_seed2/ckpt/ckpt_best.pth.tar',

        # easy env
        # rl_model_path='/home/puyuan/DI-engine/data_gfootball/gfootball_easy_dqn_seed0_rbs1e5_df0.97/ckpt/ckpt_best.pth.tar',
        # replay_path='/home/puyuan/DI-engine/data_gfootball/gfootball_easy_dqn_seed0_rbs1e5_df0.97',
        
        # medium env
        # rl_model_path='/home/puyuan/DI-engine/data_gfootball/gfootball_medium_dqn_seed0_rbs1e5_df0.999/ckpt/ckpt_best.pth.tar',
        # replay_path='/home/puyuan/DI-engine/data_gfootball/gfootball_medium_dqn_seed0_rbs1e5_df0.999',
        
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
        eval=dict(evaluator=dict(eval_freq=5000, n_episode=evaluator_env_num)),
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
    # env_manager=dict(type='subprocess'),
    env_manager=dict(type='base'),
    policy=dict(type='dqn'),
)
gfootball_dqn_create_config = EasyDict(gfootball_dqn_create_config)
create_config = gfootball_dqn_create_config

if __name__ == '__main__':
    # or you can enter `ding -m serial -c gfootball_dqn_config.py -s 0`
    from ding.entry import serial_pipeline
    from dizoo.gfootball.model.q_network.football_q_network import FootballNaiveQ
    football_naive_q = FootballNaiveQ()
    serial_pipeline((main_config, create_config), model=football_naive_q, seed=0, max_env_step=5e6)
