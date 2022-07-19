from easydict import EasyDict

collector_env_num = 8
evaluator_env_num = 5

gfootball_il_main_config = dict(
    exp_name='data_gfootball/gfootball_il_seed0',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        stop_value=999,
        env_name="11_vs_11_easy_stochastic",
        # env_name="11_vs_11_stochastic",  # default: medium
        # env_name="11_vs_11_hard_stochastic",
        save_replay_gif=False,
    ),
    policy=dict(
        continuous=False,
        test_accuracy=False,
        # Note, only if test_accuracy=True, we will test accuracy in train dataset and validation dataset
        # use the pre-trained il model in the path <il_model_path>.
        # Users should add their own il model path here. Model path should lead to a model.
        # Absolute path is recommended. In DI-engine, it is ``exp_name/ckpt/ckpt_best.pth.tar``.
        il_model_path='il_model_path_placeholder',
        cuda=True,
        model=dict(),
        learn=dict(
            update_per_collect=20,
            # batch_size=512,
            batch_size=5,
            learning_rate=0.0001,
            target_update_freq=500,
            weight_decay=1e-4,
        ),
        collect=dict(n_sample=4096, ),
        eval=dict(evaluator=dict(eval_freq=1000, n_episode=evaluator_env_num)),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.1,
                decay=10000,
            ),
            replay_buffer=dict(replay_buffer_size=int(1e6), ),
            ),
        ),
)
gfootball_il_main_config = EasyDict(gfootball_il_main_config)
main_config = gfootball_il_main_config

gfootball_il_create_config = dict(
    env=dict(
        type='gfootball',
        import_names=['dizoo.gfootball.envs.gfootball_env'],
    ),
    # env_manager=dict(type='subprocess'),
    env_manager=dict(type='base'),
    policy=dict(type='football_bc'),
)
gfootball_il_create_config = EasyDict(gfootball_il_create_config)
create_config = gfootball_il_create_config
