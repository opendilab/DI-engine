from easydict import EasyDict
from copy import deepcopy

walker2d_dt_config = dict(
    exp_name='walker2d_expert_dt_seed0',
    env=dict(
        env_id='Walker2d-v3',
        norm_obs=dict(use_norm=False, ),
        norm_reward=dict(use_norm=False, ),
        collector_env_num=1,
        evaluator_env_num=8,
        use_act_scale=True,
        n_evaluator_episode=8,
        stop_value=6000,
    ),
    policy=dict(
        env_name='Walker2d-v3',
        log_dir = './DI-engine/dizoo/d4rl/config',
        cuda=True,
        stop_value=6000,
        rtg_target=6000,  # max target return to go
        max_eval_ep_len=1000,  # max lenght of one episode
        num_eval_ep=10,  # num of evaluation episode
        batch_size=64,
        wt_decay=1e-4,
        warmup_steps=10000,
        num_updates_per_iter=100,
        context_len=20,
        n_blocks=3,
        embed_dim=128,
        n_heads=1,
        dropout_p=0.1,
        discount_factor=0.999,
        nstep=3,
        model=dict(
            state_dim=17,
            act_dim=6,
            n_blocks=3,
            h_dim=128,
            context_len=20,
            n_heads=1,
            drop_p=0.1,
            continuous=True,
        ),
        learn=dict(
            # dataset_path (:obj:`str`): The pre-collected dataset path, which should
            # point to the hdf5 file of the pre-collected dataset, and an absolute path is recommended.
            # the path is usually something like ``./walker2d_expert-v2.hdf5``.
            dataset_path='./walker2d_expert-v2.hdf5',
            learning_rate=0.0001,
            target_update_freq=100,
            kappa=1.0,
            min_q_weight=4.0,
        ),
        collect=dict(unroll_len=1, ),
        eval=dict(evaluator=dict(evalu_freq=100, ), ),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.1,
                decay=10000,
            ),
            replay_buffer=dict(replay_buffer_size=1000, ),
        ),
    ),
)

walker2d_dt_config = EasyDict(walker2d_dt_config)
main_config = walker2d_dt_config
walker2d_dt_create_config = dict(
    env=dict(
        type='mujoco',
        import_names=['dizoo.mujoco.envs.mujoco_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='dt'),
)
walker2d_dt_create_config = EasyDict(walker2d_dt_create_config)
create_config = walker2d_dt_create_config

if __name__ == "__main__":
    from ding.entry import serial_pipeline_dt
    config = deepcopy([main_config, create_config])
    serial_pipeline_dt(config, seed=0, max_train_iter=1000)
