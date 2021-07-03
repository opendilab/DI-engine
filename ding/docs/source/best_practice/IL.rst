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
`policy Overview <../feature/policy_overview.html>`__

**Use Demonstrations to IL**

DI-engine provides serial entry for IL implementation. By specify the
prepared expert data ``expert_data_path``, you can deploy IL by the
following codes:

.. code:: python

   _, converge_stop_flag = serial_pipeline_il(il_config, seed=314, data_path=expert_data_path)

**Use Expert Model to IL**

DI-engine provides data collector functions in
``ding/entry/application_entry.py``. You can give a policy configure
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

**Online IL through Seiral Pipline**

DI-engine's `serial_entry_il` provides a sub-implementation of serial pipline,
in which there is no collectors (or use collectors only to collect data at the beginning
of training). However, many IL algorithms (Dagger, SQIL, etc.) need to collect demonstration
as well as training IL model. In this case, DI-engine can use `serial_entry` to perform this
pipline. Users can define a new IL policy, the collect model of this policy is the expert
policy, and the learn model can be any supervised learning model or other IL learn model.
More details about the policy defination of DI-engine can be found in 
`policy Overview <../feature/policy_overview.html>`__
DI-engine also provide a demo of this policy in `ding/policy/il.py`. It provides a supervised 
learning pipline to imitate from an expert model online on Google Research football environments.


DI-engine provide a full demo of using PPO as both the expert policy to
generate data and the IL policy to implement behavioral cloning. You can
refer to ``ding/entry/tests/test_serial_entry_il.py`` for more
details.
