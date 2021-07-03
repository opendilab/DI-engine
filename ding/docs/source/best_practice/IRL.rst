Inverse RL
============================================
Generally, an IRL training process alternates between three stages. In
each round:

   1. New data is collected with the current RL policy. 
   2. Then a reward model is updated with the collected data in this round and the pre-collected expert data. 
   3. Finally the RL policy is trained with the current reward model.

In DI-engine, IRL is implemented with fair concise modifications on the
propcess of RL. Specifically:

   1. We push the collected data into the training set of the reward model at each return of collected data; 
   2. Train the reward model and clear the training set of the reward model before each round of RL training; 
   3. Update the rewards of each batch with the reward model before each step of RL training.


Example(GAIL)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`link <../../../entry/tests/test_serial_entry_reward_model.py>`_: 

The key code of training a DQN policy with GAIL on CartPole is provided
here:

First, create the reward model of GAIL.

.. code:: 

   from ding.irl_utils import GailRewardModel
   irl_config = {
         'input_dims': 5,
         'hidden_dims': 64,
         'batch_size': 64,
         'update_per_collect': 100,
         'expert_data_path': 'expert_data.pkl'
   	}
   reward_model = GailRewardModel(irl_config, 'cuda', tb_logger)

Here the ``GailRewardModel`` is a simple two-layer MLP, ``input_dims`` and
``hidden_dims`` are the dimension of its input layer and hidden layer. The
output layer has a dimension 1, denoting the reward. In each round the
reward model is trained by ``update_per_collect`` steps. During each step,
two batches of data, whose size are both ``batch_size``, are sampled
separately from the expert data, whose path is ``expert_data_path``, and
the collected data from the RL policy in this round.

A round of three stages are demonstrated here:

In each round of data collection, the data is pushed simutaneously into
the replay buffer and the training set of the reward model:

.. code:: python

   while new_data_count < target_new_data_count:
     eps = epsilon_greedy(collector.envstep)
     new_data = collector.collect_data(learner.train_iter, policy_kwargs={'eps': eps})
     new_data_count += len(new_data)
     # collect data for reward_model training
     reward_model.collect_data(new_data)
     replay_buffer.push(new_data, cur_collector_envstep=collector.envstep)

Then, the reward model is trained and the training set is released:

.. code:: 

   # Update reward model
   reward_model.train()
   reward_model.clear_data()

Finally, the RL policy can be trained with the reward estimated by the
reward model.

.. code:: 

       # Train the current policy from the collected data
       for i in range(cfg.policy.learn.update_per_collect):
           # Learner will train ``update_per_collect`` times in one iteration.
           train_data = replay_buffer.sample(learner.policy.get_attribute('batch_size'), learner.train_iter)
           if train_data is None:
               # It is possible that replay buffer's data count is too few to train ``update_per_collect`` times
               logging.warning(
                   "Replay buffer's data can only train for {} steps. ".format(i) +
                   "You can modify data collect config, e.g. increasing n_sample, n_episode."
               )
               break
           # Estimate the reward of the data for train
           reward_model.estimate(train_data)
           # Train the current policy with estimated reward
           learner.train(train_data, collector.envstep)


 
