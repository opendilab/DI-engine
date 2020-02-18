Supervised Learning
===================

.. toctree::
    :maxdepth: 3

Data Parse
~~~~~~~~~~
  - **pysc2.lib.features**  (observation and action interface)
  - **pysc2.lib.actions**  (action definition)
  - **pysc2.lib.action_dict**  (action info and mask)
  - **pysc2.lib.static_data**  (ability, unit type, buff and upgrade definition)
  - **sc2learner.envs.observations.alphastar_obs_wrapper**  (observation transform for alphastar)
  - **sc2learner.envs.actions.alphastar_act_wrapper**  (action transform for alphastar)
  - **sc2learner.bin.replay_decode**  (decode replay and generate offline data)
  - **sc2learner.bin.process_stat**  (process generated statistics data)

.. note::
    
  we only need to care about the code parts in pysc2 which is related to sc2learner

Dataset
~~~~~~~
  - **sc2learner.dataset.replay_dataset**  (dataset design for supervised learning from replay)

Learner
~~~~~~~
Trainer
^^^^^^^
  - **sc2learner.agents.solver.learner.sl_learner**  (base class for supervised learning)
  - **sc2learner.agents.solver.learner.alphastar_sl_learner**  (alphastar supervised learning loss and special setting)


Evaluator
^^^^^^^^^
TBD


Network
~~~~~~~
  - **sc2learner.agents.model.alphastar.obs_encoder.entity_encoder**  (entity info encoder)
  - **sc2learner.agents.model.alphastar.obs_encoder.scalar_encoder**  (global scalar info encoder)
  - **sc2learner.agents.model.alphastar.obs_encoder.spatial_encoder**  (spatial info encoder)
  - **sc2learner.agents.model.alphastar.core**  (core lstm)
  - **sc2learner.agents.model.alphastar.head.action_type_head**  (predict next action type)
  - **sc2learner.agents.model.alphastar.head.action_arg_head**  (predict next action args)
  - **sc2learner.agents.model.alphastar.policy_network**  (policy network, including all the submodules)
