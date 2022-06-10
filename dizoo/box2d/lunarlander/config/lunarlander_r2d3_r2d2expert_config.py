import os
from easydict import EasyDict

module_path = os.path.dirname(__file__)

collector_env_num = 8
evaluator_env_num = 8
expert_replay_buffer_size = int(5e3)
"""agent config"""
lunarlander_r2d3_config = dict(
    exp_name='lunarlander_r2d3_r2d2expert_seed0',
    env=dict(
        # Whether to use shared memory. Only effective if "env_manager_type" is 'subprocess'
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        env_id='LunarLander-v2',
        n_evaluator_episode=8,
        stop_value=200,
    ),
    policy=dict(
        cuda=True,
        on_policy=False,
        priority=True,
        priority_IS_weight=True,
        model=dict(
            obs_shape=8,
            action_shape=4,
            encoder_hidden_size_list=[128, 128, 512],
        ),
        discount_factor=0.997,
        nstep=5,
        burnin_step=2,
        # (int) the whole sequence length to unroll the RNN network minus
        # the timesteps of burnin part,
        # i.e., <the whole sequence length> = <unroll_len> = <burnin_step> + <learn_unroll_len>
        learn_unroll_len=40,
        learn=dict(
            # according to the r2d3 paper, actor parameter update interval is 400
            # environment timesteps, and in per collect phase, we collect 32 sequence
            # samples, the length of each samlpe sequence is <burnin_step> + <unroll_len>,
            # which is 100 in our seeting, 32*100/400=8, so we set update_per_collect=8
            # in most environments
            value_rescale=True,
            update_per_collect=8,
            batch_size=64,
            learning_rate=0.0005,
            target_update_theta=0.001,
            # DQFD related parameters
            lambda1=1.0,  # n-step return
            lambda2=1.0,  # supervised loss
            lambda3=1e-5,  # L2  it's very important to set Adam optimizer optim_type='adamw'.
            lambda_one_step_td=1,  # 1-step return
            margin_function=0.8,  # margin function in JE, here we implement this as a constant
            per_train_iter_k=0,  # TODO(pu)
        ),
        collect=dict(
            # NOTE: It is important that set key traj_len_inf=True here,
            # to make sure self._traj_len=INF in serial_sample_collector.py.
            # In R2D2 policy, for each collect_env, we want to collect data of length self._traj_len=INF
            # unless the episode enters the 'done' state.
            # In each collect phase, we collect a total of <n_sample> sequence samples.
            n_sample=32,
            traj_len_inf=True,
            env_num=collector_env_num,
            # The hyperparameter pho, the demo ratio, control the propotion of data coming\
            # from expert demonstrations versus from the agent's own experience.
            pho=1 / 4,  # TODO(pu)
        ),
        eval=dict(env_num=evaluator_env_num, ),
        other=dict(
            eps=dict(
                type='exp',
                start=0.95,
                end=0.1,
                decay=100000,
            ),
            replay_buffer=dict(
                replay_buffer_size=int(1e4),
                # (Float type) How much prioritization is used: 0 means no prioritization while 1 means full prioritization
                alpha=0.6,  # priority exponent default=0.6
                # (Float type)  How much correction is used: 0 means no correction while 1 means full correction
                beta=0.4,
            )
        ),
    ),
)
lunarlander_r2d3_config = EasyDict(lunarlander_r2d3_config)
main_config = lunarlander_r2d3_config
lunarlander_r2d3_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['dizoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='r2d3'),
)
lunarlander_r2d3_create_config = EasyDict(lunarlander_r2d3_create_config)
create_config = lunarlander_r2d3_create_config
"""export config"""
expert_lunarlander_r2d3_config = dict(
    exp_name='expert_lunarlander_r2d3_r2d2expert_seed0',
    env=dict(
        # Whether to use shared memory. Only effective if "env_manager_type" is 'subprocess'
        collector_env_num=collector_env_num,
        evaluator_env_num=evaluator_env_num,
        n_evaluator_episode=5,
        stop_value=200,
    ),
    policy=dict(
        cuda=True,
        on_policy=False,
        priority=True,
        model=dict(
            obs_shape=8,
            action_shape=4,
            encoder_hidden_size_list=[128, 128, 512],  # r2d2
        ),
        discount_factor=0.997,
        burnin_step=2,
        nstep=5,
        learn=dict(expert_replay_buffer_size=expert_replay_buffer_size, ),
        collect=dict(
            # NOTE: It is important that set key traj_len_inf=True here,
            # to make sure self._traj_len=INF in serial_sample_collector.py.
            # In R2D2 policy, for each collect_env, we want to collect data of length self._traj_len=INF
            # unless the episode enters the 'done' state.
            # In each collect phase, we collect a total of <n_sample> sequence samples.
            n_sample=32,
            traj_len_inf=True,
            # Users should add their own model path here. Model path should lead to a model.
            # Absolute path is recommended.
            # In DI-engine, it is ``exp_name/ckpt/ckpt_best.pth.tar``.
            model_path='model_path_placeholder',
            # Cut trajectories into pieces with length "unroll_len",
            # which should set as self._sequence_len of r2d2
            unroll_len=42,  # NOTE: should equals self._sequence_len in r2d2 policy
            env_num=collector_env_num,
        ),
        eval=dict(env_num=evaluator_env_num, ),
        other=dict(
            replay_buffer=dict(
                replay_buffer_size=expert_replay_buffer_size,
                # (Float type) How much prioritization is used: 0 means no prioritization while 1 means full prioritization
                alpha=0.9,  # priority exponent default=0.6
                # (Float type)  How much correction is used: 0 means no correction while 1 means full correction
                beta=0.4,
            )
        ),
    ),
)
expert_lunarlander_r2d3_config = EasyDict(expert_lunarlander_r2d3_config)
expert_main_config = expert_lunarlander_r2d3_config
expert_lunarlander_r2d3_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['dizoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='r2d2_collect_traj'),  # this policy is designed to collect r2d2 expert traj for r2d3
)
expert_lunarlander_r2d3_create_config = EasyDict(expert_lunarlander_r2d3_create_config)
expert_create_config = expert_lunarlander_r2d3_create_config

if __name__ == "__main__":
    from ding.entry import serial_pipeline_r2d3
    serial_pipeline_r2d3([main_config, create_config], [expert_main_config, expert_create_config], seed=0)
