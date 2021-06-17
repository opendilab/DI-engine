How to use multiple buffers?
================================================


Guideline
^^^^^^^^^^^^^^

In some algorithms, it is required to use multiple buffers. For example, in `R2D3 <https://arxiv.org/pdf/1909.01387.pdf>`_, there is a demonstration buffer full of expert data and an agent buffer collected during training; In `Phasic Policy Gradient <https://arxiv.org/pdf/2009.04416.pdf>`_, there is a value buffer and a policy buffer respectively designed for critic optimization and actor optimization.

However, nerveX `serial_pipeline` only supports single buffer. So in this section, we will teach you how to write the configuration file and your own multi-buffer pipeline. We will take `PPG` algorithm as an example.

1. Config
^^^^^^^^^^^^^^
    
    We show core part of `nerveX/app_zoo/classic_control/cartpole/config/cartpole_ppg_config.py` as follows:

    .. code:: python
        
        cartpole_ppg_config = dict(
            # ...
            policy=dict(
                # ...
                other=dict(
                    replay_buffer=dict(
                        multi_buffer=True,
                        policy=dict(
                            replay_buffer_size=100,
                            max_use=10,
                        ),
                        value=dict(
                            replay_buffer_size=1000,
                            max_use=100,
                        ),
                    ),
                ),
            ),
        )
    
    You must set ``multi_buffer`` to True; Then list all buffers' configuration.

    If you want to use `create_cfg`, you must list all buffers' type:

    .. code:: python

        cartpole_ppg_create_config = dict(
            # ...
            replay_buffer=dict(
                policy=dict(type='priority'),
                value=dict(type='priority'),
            )
        )

2. Pipeline
^^^^^^^^^^^^^^

    After finishing the config file, you can write your own pipeline. Here is an example from `app_zoo/classic_control/cartpole/entry/cartpole_ppg_main.py`.

    .. code:: python

        # Init two buffers
        policy_buffer = AdvancedReplayBuffer(cfg.policy.other.replay_buffer.policy, tb_logger, 'policy')
        value_buffer = AdvancedReplayBuffer(cfg.policy.other.replay_buffer.value, tb_logger, 'value')

        while True:
            # ...
            new_data = collector.collect(train_iter=learner.train_iter)
            # Push data into two buffers repectively. Remember to deepcopy if it is a must.
            policy_buffer.push(new_data, cur_collector_envstep=collector.envstep)
            value_buffer.push(deepcopy(new_data), cur_collector_envstep=collector.envstep)
            for i in range(cfg.policy.learn.update_per_collect):
                batch_size = learner.policy.get_attribute('batch_size')
                # Sample from two buffers respectively. Form the new `train_data` and start learner training.
                policy_data = policy_buffer.sample(batch_size['policy'], learner.train_iter)
                value_data = policy_buffer.sample(batch_size['value'], learner.train_iter)
                if policy_data is not None and value_data is not None:
                    train_data = {'policy': policy_data, 'value': value_data}
                    learner.train(train_data, collector.envstep)
            # Since `PPG` is on-policy, remember to clear two buffers.
            policy_buffer.clear()
            value_buffer.clear()