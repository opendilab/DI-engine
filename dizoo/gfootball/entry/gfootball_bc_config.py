"""
Overview:
    Here is the behaviour cloning (BC) default config for gfootball.
    For main entry, please refer to the gfootball_bc_rule_main.py,
    gfootball_bc_rule_lt0_main.py, gfootball_bc_kaggle5th_main.py in the same directory.
"""
from easydict import EasyDict

collector_env_num = 8
evaluator_env_num = 5

gfootball_bc_config = dict(
    exp_name='gfootball_bc_seed0',
    env=dict(
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=evaluator_env_num,
        stop_value=999,  # Don't stop until training <train_epoch> epochs
        env_name="11_vs_11_easy_stochastic",
        # env_name="11_vs_11_stochastic",  # default: medium
        # env_name="11_vs_11_hard_stochastic",
        save_replay_gif=False,
        manager=dict(shared_memory=False, ),
    ),
    policy=dict(
        env_name='gfootball',
        continuous=False,
        # action_shape is effective only when continuous=False
        action_shape=19,
        show_train_test_accuracy=False,
        # Note, only if show_train_test_accuracy=True, we will test accuracy in train dataset and validation dataset
        # use the pre-trained BC model in the path <bc_model_path>.
        # Users should add their own BC model path here. Model path should lead to a model.
        # Absolute path is recommended. In DI-engine, it is ``exp_name/ckpt/ckpt_best.pth.tar``.
        bc_model_path='bc_model_path_placeholder',
        cuda=True,
        model=dict(),
        learn=dict(
            update_per_collect=20,
            batch_size=512,
            learning_rate=0.0001,
            target_update_freq=500,
            weight_decay=None,
            ce_label_smooth=False,
            show_accuracy=False,
        ),
        collect=dict(n_sample=4096, ),
        eval=dict(evaluator=dict(eval_freq=1000)),
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
gfootball_bc_config = EasyDict(gfootball_bc_config)
main_config = gfootball_bc_config

gfootball_bc_create_config = dict(
    env=dict(
        type='gfootball',
        import_names=['dizoo.gfootball.envs.gfootball_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='bc'),
)
gfootball_bc_create_config = EasyDict(gfootball_bc_create_config)
create_config = gfootball_bc_create_config
