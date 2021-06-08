Imitation Learning
====================

Guideline
~~~~~~~~~~~~

In some environments where the rewards are sparse (e.g. a game where we
only receive a reward when the game is won or lost), the normal RL
approach can be very struggle. A feasible solution to this problem is
imitation learning (IL). In IL instead of trying to learn from the
sparse rewards or manually specifying a reward function, an expert
(typically a human) provides us with a set of demonstrations. The agent
then tries to learn the optimal policy by following, imitating the
expert’s decisions.

The simplest idea of imitation learning is behavioral cloning(BC). BC
can be described as the following steps:

-  Collect demonstrations (:math:`\tau^{*}` trajectories) from expert

-  Treat the demonstrations as i.i.d state-action pairs:
   :math:`(s_0^*,a_0^*),(s_1^*,a_1^*),...`

-  Learn :math:`\pi_{\theta}` policy using supervised learning by
   minimizing the loss function :math:`L(a^*,\pi_{\theta}(s))`

Behavioral cloning can be quite problematic. The main reason for this is
the i.i.d. assumption: while supervised learning assumes that the
state-action pairs are distributed i.i.d., in MDP an action in a given
state induces the next state, which breaks the previous assumption. This
also means, that errors made in different states add up, therefore a
mistake made by the agent can easily put it into a state that the expert
has never visited and the agent has never trained on. In such states,
the behavior is undefined and this can lead to catastrophic failures.

For the majority of the cases, behavioral cloning can be quite
problematic. But due to the clarity of behavioral cloning, our demo of
imitation learning will be given in BC.

Demo
~~~~~

You can use either expert model or demonstrations to perform imitation
learning. Usually you need to define an imitation learning policy. For
policy registration, you can refer to
`policy <../feature/policy_overvies.html>`__

**Use Demonstrations to IL**

NerveX provides serial entry for IL implementation. By specify the
prepared expert data ``expert_data_path``, you can deploy IL by the
following codes:

.. code:: python

   _, converge_stop_flag = serial_pipeline_il(il_config, seed=314, data_path=expert_data_path)

**Use Expert Model to IL**

NerveX provides data collector functions in
``nervex/entry/application_entry.py``. You can give a policy configure
to train an RL model from scratch(or load an existing model), then use
the model to generate data for IL. This pipline can be describe as the
following codes:

.. code:: python

   expert_policy = serial_pipeline(train_config, seed=0)

   # collect expert demo data
   collect_count = 10000
   expert_data_path = 'expert_data.pkl'
   state_dict = expert_policy.collect_mode.state_dict()
   collect_config = [deepcopy(cartpole_ppo_config), deepcopy(cartpole_ppo_create_config)]
   collect_demo_data(
       collect_config, seed=0, state_dict=state_dict, expert_data_path=expert_data_path, collect_count=collect_count
   )
   # il training
   _, converge_stop_flag = serial_pipeline_il(il_config, seed=314, data_path=expert_data_path)

NerveX provide a full demo of using PPO as both the expert policy to
generate data and the IL policy to implement behavioral cloning. You can
refer to ``nervex/entry/tests/test_serial_entry_il.py`` for more
details.
