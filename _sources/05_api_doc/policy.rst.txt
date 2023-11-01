ding.policy
-----------

Base Policy
=============
Please refer to ``ding/policy/base_policy.py`` for more details.

Policy
~~~~~~~

.. autoclass:: ding.policy.Policy
    :members: default_config, __init__, _init_learn, _forward_learn, _init_collect, _forward_collect, _get_train_sample, _process_transition, _init_eval, _forward_eval, _state_dict_learn, _load_state_dict_learn, default_model, _monitor_vars_learn, _set_attribute, _get_attribute, __repr__, sync_gradients, _create_model, _init_multi_gpu_setting, _state_dict_collect, _load_state_dict_collect, _state_dict_eval, _load_state_dict_eval, _reset_learn, _reset_collect, _reset_eval, learn_mode, collect_mode, eval_mode

CommandModePolicy
~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.policy.CommandModePolicy
    :members: command_mode, _init_command, _get_setting_learn, _get_setting_collect, _get_setting_eval


create_policy
~~~~~~~~~~~~~~

.. autofunction:: ding.policy.create_policy

get_policy_cls
~~~~~~~~~~~~~~

.. autofunction:: ding.policy.get_policy_cls


DQN
======
Please refer to ``ding/policy/dqn.py`` for more details.


DQNPolicy
~~~~~~~~~~~

.. autoclass:: ding.policy.DQNPolicy
    :members: _init_learn, _forward_learn, _init_collect, _forward_collect, _get_train_sample, _process_transition, _init_eval, _forward_eval, _state_dict_learn, _load_state_dict_learn, default_model, _monitor_vars_learn

DQNSTDIMPolicy
~~~~~~~~~~~~~~~

.. autoclass:: ding.policy.DQNSTDIMPolicy
    :members: _init_learn, _forward_learn, _state_dict_learn, _load_state_dict_learn, _monitor_vars_learn, _model_encode


PPO
====
Please refer to ``ding/policy/ppo.py`` for more details.

PPOPolicy
~~~~~~~~~~

.. autoclass:: ding.policy.PPOPolicy
    :members: _init_learn, _forward_learn, _init_collect, _forward_collect, _get_train_sample, _process_transition, _init_eval, _forward_eval, default_model, _monitor_vars_learn


PPOPGPolicy
~~~~~~~~~~~

.. autoclass:: ding.policy.PPOPGPolicy
    :members: _init_learn, _forward_learn, _init_collect, _forward_collect, _get_train_sample, _process_transition, _init_eval, _forward_eval, default_model, _monitor_vars_learn

PPOOffPolicy
~~~~~~~~~~~~

.. autoclass:: ding.policy.PPOOffPolicy
    :members: _init_learn, _forward_learn, _init_collect, _forward_collect, _get_train_sample, _process_transition, _init_eval, _forward_eval, default_model, _monitor_vars_learn

PPOSTDIMPolicy
~~~~~~~~~~~~~~

.. autoclass:: ding.policy.PPOSTDIMPolicy
    :members: _init_learn, _forward_learn, _state_dict_learn, _load_state_dict_learn, _monitor_vars_learn, _model_encode

BC
===
Please refer to ``ding/policy/bc.py`` for more details.

BehaviourCloningPolicy
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.policy.BehaviourCloningPolicy
    :members: _init_learn, _forward_learn, _init_eval, _forward_eval, default_model, _monitor_vars_learn, _init_collect


DDPG
=====
Please refer to ``ding/policy/ddpg.py`` for more details.

DDPGPolicy
~~~~~~~~~~

.. autoclass:: ding.policy.DDPGPolicy
    :members: _init_learn, _forward_learn, _init_collect, _forward_collect, _get_train_sample, _process_transition, _init_eval, _forward_eval, default_model, _state_dict_learn, _load_state_dict_learn, _monitor_vars_learn

TD3
===
Please refer to ``ding/policy/td3.py`` for more details.

TD3Policy
~~~~~~~~~~

.. autoclass:: ding.policy.TD3Policy
    :members: _init_learn, _forward_learn, _init_collect, _forward_collect, _get_train_sample, _process_transition, _init_eval, _forward_eval, default_model, _state_dict_learn, _load_state_dict_learn, _monitor_vars_learn

SAC
====

Please refer to ``ding/policy/sac.py`` for more details.

SACPolicy
~~~~~~~~~~

.. autoclass:: ding.policy.SACPolicy
    :members: _init_learn, _forward_learn, _init_collect, _forward_collect, _get_train_sample, _process_transition, _init_eval, _forward_eval, default_model, _state_dict_learn, _load_state_dict_learn, _monitor_vars_learn

DiscreteSACPolicy
~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.policy.DiscreteSACPolicy
    :members: _init_learn, _forward_learn, _init_collect, _forward_collect, _get_train_sample, _process_transition, _init_eval, _forward_eval, _state_dict_learn, _load_state_dict_learn, default_model, _monitor_vars_learn

SQILSACPolicy
~~~~~~~~~~~~~~

.. autoclass:: ding.policy.SQILSACPolicy
    :members: _init_learn, _forward_learn, _monitor_vars_learn 

R2D2
=====
Please refer to ``ding/policy/r2d2.py`` for more details.

R2D2Policy
~~~~~~~~~~~

.. autoclass:: ding.policy.R2D2Policy
    :members: _init_learn, _forward_learn, _init_collect, _forward_collect, _get_train_sample, _process_transition, _state_dict_learn, _load_state_dict_learn, default_model, _monitor_vars_learn, _reset_learn, _reset_eval, _reset_collect

IMPALA
======
Please refer to ``ding/policy/impala.py`` for more details.

IMPALAPolicy
~~~~~~~~~~~~

.. autoclass:: ding.policy.IMPALAPolicy
    :members: _init_learn, _forward_learn, _init_collect, _forward_collect, _get_train_sample, _init_eval, _forward_eval, _process_transition, default_model, _monitor_vars_learn

QMIX
=====

Please refer to ``ding/policy/qmix.py`` for more details.

QMIXPolicy
~~~~~~~~~~

.. autoclass:: ding.policy.QMIXPolicy
    :members: _init_learn, _forward_learn, _init_collect, _forward_collect, _get_train_sample, _process_transition, _state_dict_learn, _load_state_dict_learn, default_model, _monitor_vars_learn, _reset_learn, _reset_eval, _reset_collect

CQL
====
Please refer to ``ding/policy/cql.py`` for more details.

CQLPolicy
~~~~~~~~~

.. autoclass:: ding.policy.CQLPolicy
    :members: _init_learn, _forward_learn

DiscreteCQLPolicy
~~~~~~~~~~~~~~~~~~~

.. autoclass:: ding.policy.DiscreteCQLPolicy
    :members: _init_learn, _forward_learn, _monitor_vars_learn


DecisionTransformer
===================

Please refer to ``ding/policy/dt.py`` for more details.

DTPolicy
~~~~~~~~

.. autoclass:: ding.policy.DTPolicy
    :members: _init_learn, _forward_learn, _init_eval, _forward_eval, _reset_eval, _monitor_vars_learn

PDQN
====

Please refer to ``ding/policy/pdqn.py`` for more details.

PDQNPolicy
~~~~~~~~~~

.. autoclass:: ding.policy.PDQNPolicy
    :members: _init_learn, _forward_learn, _init_collect, _forward_collect, _init_eval, _forward_eval, _get_train_sample, _process_transition, _state_dict_learn, _load_state_dict_learn, default_model, _monitor_vars_learn


MDQN
======

Please refer to ``ding/policy/mdqn.py`` for more details.

MDQNPolicy
~~~~~~~~~~

.. autoclass:: ding.policy.MDQNPolicy
    :members: _init_learn, _forward_learn, _monitor_vars_learn 


Policy Factory
==============
Please refer to ``ding/policy/policy_factory.py`` for more details.


PolicyFactory
~~~~~~~~~~~~~

.. autoclass:: ding.policy.PolicyFactory
   :members: get_random_policy

get_random_policy
~~~~~~~~~~~~~~~~~
.. autofunction:: ding.policy.get_random_policy


Common Utilities
================
Please refer to ``ding/policy/common_utils.py`` for more details.


default_preprocess_learn
~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.policy.default_preprocess_learn

single_env_forward_wrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: ding.policy.single_env_forward_wrapper

single_env_forward_wrapper_ttorch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: ding.policy.single_env_forward_wrapper_ttorch
