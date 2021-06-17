How to randomly collect some data sample at the beginning?
==============================================================


Guideline
^^^^^^^^^^^^^^
For some policies and environments, it is better to collect some data samples at the very beginning, with a completely random policy.
So in this section, we will introduce how to write config and env info, and how serial_pipeline randomly collect data.

1. Config
^^^^^^^^^^^^^
    Specify how many data samples to collect at the beginning:

    .. code:: python

        cartpole_rainbow_config = dict(
            # ...
            random_collect_size=2000,
        )

2. Make sure action space is an available attribute in env
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    Env manager will get env_info from an env_ref, so you must make sure that `act_space` is available in env's ``info`` method.

    .. code:: python

        def info(self) -> BaseEnvInfo:
            T = EnvElementInfo
            return BaseEnvInfo(
                # ...
                # [min, max)
                act_space=T(
                    (1, ),
                    {
                        'min': 0,
                        'max': 2,
                        'dtype': int,
                    },
                ),
                # ...
            )

3. Randomly collect data at the beginning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    nerveX `serial_pipeline` will collect data if `random_collect_size` is set in config.

    .. code:: python

        # Accumulate plenty of data at the beginning of training.
        if cfg.policy.get('random_collect_size', 0) > 0:
            action_space = collector_env.env_info().act_space
            # action_space is used to init random_policy
            random_policy = PolicyFactory.get_random_policy(policy.collect_mode, action_space=action_space)
            # Reset collector's policy to random_policy
            collector.reset_policy(random_policy)
            collect_kwargs = commander.step()
            # Randomly collect data and push into buffer
            new_data = collector.collect(n_sample=cfg.policy.random_collect_size, policy_kwargs=collect_kwargs)
            replay_buffer.push(new_data, cur_collector_envstep=0)
            # Change collector's policy back to the collect_mode policy
            collector.reset_policy(policy.collect_mode)

You can refer to policy `rainbow` as an example.