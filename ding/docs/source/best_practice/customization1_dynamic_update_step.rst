Customization 1: Dynamic Update Step
=====================================

The update iterations per collection is set by default in the ``update_per_collect`` in config. In some cases, you can dynamically modify
this value in each collection to balance the training & collection ratio. An example is shown as follows.

.. code-block:: python

        new_data = collector.collect_data(learner.train_iter)
        replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)
        # update each sample 3 times on average
        update_per_collect = len(new_data) * 3 // learner.policy.learn.batch_size
        for i in range(update_per_collect):
            train_data = replay_buffer.sample(learner.policy.learn.batch_size, learner.train_iter)
            if train_data is not None:
                learner.train(train_data, collector.envstep)
