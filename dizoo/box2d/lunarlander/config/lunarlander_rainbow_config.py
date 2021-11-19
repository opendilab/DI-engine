from easydict import EasyDict
from ding.entry import serial_pipeline

nstep = 3
lunarlander_rainbow_default_config = dict(
    env=dict(
        # Whether to use shared memory. Only effective if "env_manager_type" is 'subprocess'
        manager=dict(shared_memory=True, ),
        # Env number respectively for collector and evaluator.
        collector_env_num=8,
        evaluator_env_num=5,
        n_evaluator_episode=5,
        stop_value=5,
    ),
    policy = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        # type='rainbow',
        # (bool) Whether to use cuda for network.
        cuda=False,
        # (bool) Whether the RL algorithm is on-policy or off-policy.
        on_policy=False,
        # (bool) Whether use priority(priority sample, IS weight, update priority)
        priority=True,
        # (bool) Whether use Importance Sampling Weight to correct biased update. If True, priority must be True.
        priority_IS_weight=True,
        # (int) Number of training samples(randomly collected) in replay buffer when training starts.
        # random_collect_size=2000,
        model=dict(
            #####
            obs_shape=8,
            action_shape=4,
            encoder_hidden_size_list=[512, 64],
            #####
            # (float) Value of the smallest atom in the support set.
            # Default to -10.0.
            v_min=-10,
            # (float) Value of the smallest atom in the support set.
            # Default to 10.0.
            v_max=10,
            # (int) Number of atoms in the support set of the
            # value distribution. Default to 51.
            n_atom=51,
        ),
        # (float) Reward's future discount factor, aka. gamma.
        discount_factor=0.99,
        # (int) N-step reward for target q_value estimation
        nstep=3,
        learn=dict(
            # (bool) Whether to use multi gpu
            multi_gpu=False,
            # How many updates(iterations) to train after collector's one collection.
            # Bigger "update_per_collect" means bigger off-policy.
            # collect data -> update policy-> collect data -> ...
            update_per_collect=1,
            batch_size=32,
            learning_rate=0.001,
            # ==============================================================
            # The following configs are algorithm-specific
            # ==============================================================
            # (int) Frequence of target network update.
            target_update_freq=100,
            # (bool) Whether ignore done(usually for max step termination env)
            ignore_done=False,
        ),
        # collect_mode config
        collect=dict(
            # (int) Only one of [n_sample, n_episode] shoule be set
            n_sample=32,
            # (int) Cut trajectories into pieces with length "unroll_len".
            unroll_len=1,
        ),
        eval=dict(),
        # other config
        other=dict(
            # Epsilon greedy with decay.
            eps=dict(
                # (str) Decay type. Support ['exp', 'linear'].
                type='exp',
                # (float) End value for epsilon decay, in [0, 1]. It's equals to `end` because rainbow uses noisy net.
                start=0.05,
                # (float) End value for epsilon decay, in [0, 1].
                end=0.05,
                # (int) Env steps of epsilon decay.
                decay=100000,
            ),
            replay_buffer=dict(
                # (int) Max size of replay buffer.
                replay_buffer_size=100000,
                # (float) Prioritization exponent.
                alpha=0.6,
                # (float) Importance sample soft coefficient.
                # 0 means no correction, while 1 means full correction
                beta=0.4,
                # (int) Anneal step for beta: 0 means no annealing. Defaults to 0
                anneal_step=100000,
            )
        ),
    )
)
lunarlander_rainbow_default_config = EasyDict(lunarlander_rainbow_default_config)
main_config = lunarlander_rainbow_default_config

lunarlander_rainbow_create_config = dict(
    env=dict(
        type='lunarlander',
        import_names=['dizoo.box2d.lunarlander.envs.lunarlander_env'],
    ),
    env_manager=dict(type='subprocess'),
    policy=dict(type='rainbow'),
)
lunarlander_rainbow_create_config = EasyDict(lunarlander_rainbow_create_config)
create_config = lunarlander_rainbow_create_config

if __name__ == "__main__":
    serial_pipeline([main_config, create_config], seed=0)
