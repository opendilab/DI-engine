How to randomly collect some data sample at the beginning?
==============================================================


Guideline
^^^^^^^^^^^^^^
For some policies and environments, it is better to collect some data samples at the very beginning, with a completely random policy.
So in this section, we will introduce how to write config and env info, and how serial_pipeline randomly collect data.

How to write config and env info. (User View)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Config

    Specify how many data samples to collect at the beginning:

    .. code:: python

        cartpole_rainbow_config = dict(
            # ...
            policy=dict(
                # ...
                random_collect_size=2000,
            ),
        )

2. Make sure action space is an available attribute in env

    Env manager will get env_info from an env_ref, so you must make sure that `act_space` is available in env's ``info`` method.

    .. code:: python

        def info(self) -> BaseEnvInfo:
            T = EnvElementInfo
            return BaseEnvInfo(
                # Discrete action
                act_space=T(
                    (1, ),
                    {
                        # [min, max)
                        'min': 0,
                        'max': 2,
                        'dtype': int,
                    },
                ),
                # ...
                # Continuous action
                act_space=T(
                    (3, ),
                    {
                        'min': 0.,
                        'max': 1.,
                        'dtype': np.float32,
                    },
                ),
            )


How DI-engine randomly collect? (Developer View)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    We will take DI-engine ``serial_pipeline`` as an example, to demonstrate how to use random_policy(``ding/ding/policy/policy_factory.py``) to collect random data if ``random_collect_size`` is set in config.

    .. code:: python

        # Accumulate plenty of data at the beginning of training.
        if cfg.policy.get('random_collect_size', 0) > 0:
            # Acquire action space from env.
            action_space = collector_env.env_info().act_space
            # `action_space` is used by random_policy to generate legal actions.
            random_policy = PolicyFactory.get_random_policy(policy.collect_mode, action_space=action_space)
            # Reset collector's policy to random_policy
            collector.reset_policy(random_policy)
            # Randomly collect data and push them into buffer
            new_data = collector.collect(n_sample=cfg.policy.random_collect_size, policy_kwargs=collect_kwargs)
            replay_buffer.push(new_data, cur_collector_envstep=0)
            # Switch collector's policy back to the collect_mode policy
            collector.reset_policy(policy.collect_mode)
    
    DI-engine use different methods to generate different types of actions (discrete and continuous, whether has upper or lower bound, etc.).

    You can refer to ``ding/ding/policy/policy_factory.py``, see ``PolicyFactory``'s ``get_random_policy``'s ``forward`` for more details.