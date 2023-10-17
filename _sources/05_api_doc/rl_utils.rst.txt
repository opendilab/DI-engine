ding.rl_utils
--------------

a2c
========
Please refer to ``ding/rl_utils/a2c`` for more details.

a2c_error
~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.a2c_error

a2c_error_continuous
~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.a2c_error_continuous

acer
========
Please refer to ``ding/rl_utils/acer`` for more details.


acer_policy_error
~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.acer_policy_error

acer_value_error
~~~~~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.acer_value_error

acer_trust_region_update
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.acer_trust_region_update

adder
========
Please refer to ``ding/rl_utils/adder`` for more details.

Adder
~~~~~~~~~~~~~
.. autoclass:: ding.rl_utils.adder.Adder
    :members: get_gae, get_gae_with_default_last_value, get_nstep_return_data, get_train_sample, _get_null_transition

get_gae
~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.adder.get_gae

get_gae_with_default_last_value
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.adder.get_gae_with_default_last_value

get_nstep_return_data
~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.adder.get_nstep_return_data

get_train_sample
~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.adder.get_train_sample

beta_function
==============
Please refer to ``ding/rl_utils/beta_function`` for more details.

cpw
~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.beta_function.cpw

CVaR
~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.beta_function.CVaR

beta_function_map
~~~~~~~~~~~~~~~~~
.. autoattribute:: ding.rl_utils.beta_function_map

coma
========
Please refer to ``ding/rl_utils/coma`` for more details.

coma_error
~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.coma_error


exploration
============
Please refer to ``ding/rl_utils/exploration`` for more details.

get_epsilon_greedy_fn
~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.exploration.get_epsilon_greedy_fn

BaseNoise
~~~~~~~~~~~~~
.. autoclass:: ding.rl_utils.exploration.BaseNoise
    :members: __init__, __call__

GaussianNoise
~~~~~~~~~~~~~
.. autoclass:: ding.rl_utils.exploration.GaussianNoise
    :members: __init__, __call__

OUNoise
~~~~~~~~~~~~~
.. autoclass:: ding.rl_utils.exploration.OUNoise
   :members:

noise_mapping
~~~~~~~~~~~~~
.. autoattribute:: ding.rl_utils.exploration.noise_mapping

create_noise_generator
~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.exploration.create_noise_generator

gae
========
Please refer to ``ding/rl_utils/gae`` for more details.

gae_data
~~~~~~~~~~~~~
.. autoclass:: ding.rl_utils.gae.gae_data

shape_fn_gae
~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.gae.shape_fn_gae

gae
~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.gae.gae

isw
========
Please refer to ``ding/rl_utils/isw`` for more details.

compute_importance_weights
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.isw.compute_importance_weights

ppg
========
Please refer to ``ding/rl_utils/ppg`` for more details.

ppg_data
~~~~~~~~~~~~~
.. autoclass:: ding.rl_utils.ppg.ppg_data

ppg_joint_loss
~~~~~~~~~~~~~~~
.. autoclass:: ding.rl_utils.ppg.ppg_joint_loss

ppg_joint_error
~~~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.ppg.ppg_joint_error

ppo
========
Please refer to ``ding/rl_utils/ppo`` for more details.

ppo_data
~~~~~~~~~~~~~
.. autoclass:: ding.rl_utils.ppo.ppo_data

ppo_policy_data
~~~~~~~~~~~~~~~
.. autoclass:: ding.rl_utils.ppo.ppo_policy_data

ppo_value_data
~~~~~~~~~~~~~~~
.. autoclass:: ding.rl_utils.ppo.ppo_value_data
  :noindex:

ppo_loss
~~~~~~~~~~~~~
.. autoclass:: ding.rl_utils.ppo.ppo_loss

ppo_policy_loss
~~~~~~~~~~~~~~~~~
.. autoclass:: ding.rl_utils.ppo.ppo_policy_loss

ppo_info
~~~~~~~~~~~~~
.. autoclass:: ding.rl_utils.ppo.ppo_info

shape_fn_ppo
~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.ppo.shape_fn_ppo

ppo_error
~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.ppo.ppo_error

ppo_policy_error
~~~~~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.ppo.ppo_policy_error
  :noindex:

ppo_value_error
~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.ppo.ppo_value_error
  :noindex:

ppo_error_continuous
~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.ppo.ppo_error_continuous

ppo_policy_error_continuous
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.ppo.ppo_policy_error_continuous

retrace
========
Please refer to ``ding/rl_utils/retrace`` for more details.

compute_q_retraces
~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.retrace.compute_q_retraces

sampler
========
Please refer to ``ding/rl_utils/sampler`` for more details.

ArgmaxSampler
~~~~~~~~~~~~~
.. autoclass:: ding.rl_utils.sampler.ArgmaxSampler
    :members:

MultinomialSampler
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.rl_utils.sampler.MultinomialSampler
    :members:

MuSampler
~~~~~~~~~~~~~
.. autoclass:: ding.rl_utils.sampler.MuSampler
    :members:

ReparameterizationSampler
~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.rl_utils.sampler.ReparameterizationSampler
    :members:

HybridStochasticSampler
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.rl_utils.sampler.HybridStochasticSampler
    :members:

HybridDeterminsticSampler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.rl_utils.sampler.HybridDeterminsticSampler
    :members:

td
========
Please refer to ``ding/rl_utils/td`` for more details.

q_1step_td_data
~~~~~~~~~~~~~~~~
.. autoclass:: ding.rl_utils.td.q_1step_td_data

q_1step_td_error
~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.td.q_1step_td_error

m_q_1step_td_data
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.rl_utils.td.m_q_1step_td_data

m_q_1step_td_error
~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.td.m_q_1step_td_error

q_v_1step_td_data
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.rl_utils.td.q_v_1step_td_data

q_v_1step_td_error
~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.td.q_v_1step_td_error

nstep_return_data
~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.rl_utils.td.nstep_return_data

nstep_return
~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.td.nstep_return

dist_1step_td_data
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.rl_utils.td.dist_1step_td_data

dist_1step_td_error
~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.td.dist_1step_td_error

dist_nstep_td_data
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.rl_utils.td.dist_nstep_td_data

shape_fn_dntd
~~~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.td.shape_fn_dntd

dist_nstep_td_error
~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.td.dist_nstep_td_error

v_1step_td_data
~~~~~~~~~~~~~~~~
.. autoclass:: ding.rl_utils.td.v_1step_td_data

v_1step_td_error
~~~~~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.td.v_1step_td_error

v_nstep_td_data
~~~~~~~~~~~~~~~~
.. autoclass:: ding.rl_utils.td.v_nstep_td_data

v_nstep_td_error
~~~~~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.td.v_nstep_td_error

q_nstep_td_data
~~~~~~~~~~~~~~~~
.. autoclass:: ding.rl_utils.td.q_nstep_td_data

dqfd_nstep_td_data
~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.rl_utils.td.dqfd_nstep_td_data

shape_fn_qntd
~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.td.shape_fn_qntd

q_nstep_td_error
~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.td.q_nstep_td_error

bdq_nstep_td_error
~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.td.bdq_nstep_td_error

shape_fn_qntd_rescale
~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.td.shape_fn_qntd_rescale

q_nstep_td_error_with_rescale
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.td.q_nstep_td_error_with_rescale

dqfd_nstep_td_error
~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.td.dqfd_nstep_td_error

dqfd_nstep_td_error_with_rescale
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.td.dqfd_nstep_td_error_with_rescale

qrdqn_nstep_td_data
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.rl_utils.td.qrdqn_nstep_td_data

qrdqn_nstep_td_error
~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.td.qrdqn_nstep_td_error

q_nstep_sql_td_error
~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.td.q_nstep_sql_td_error

iqn_nstep_td_data
~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.rl_utils.td.iqn_nstep_td_data

iqn_nstep_td_error
~~~~~~~~~~~~~~~~~~~``
.. autofunction:: ding.rl_utils.td.iqn_nstep_td_error

fqf_nstep_td_data
~~~~~~~~~~~~~~~~~~
.. autoclass:: ding.rl_utils.td.fqf_nstep_td_data

fqf_nstep_td_error
~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.td.fqf_nstep_td_error

evaluate_quantile_at_action
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.td.evaluate_quantile_at_action

fqf_calculate_fraction_loss
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.td.fqf_calculate_fraction_loss

td_lambda_data
~~~~~~~~~~~~~~~
.. autoclass:: ding.rl_utils.td.td_lambda_data

shape_fn_td_lambda
~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.td.shape_fn_td_lambda

td_lambda_error
~~~~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.td.td_lambda_error

generalized_lambda_returns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.td.generalized_lambda_returns

multistep_forward_view
~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.td.multistep_forward_view

upgo
========
Please refer to ``ding/rl_utils/upgo`` for more details.

upgo_returns
~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.upgo.upgo_returns

upgo_loss
~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.upgo.upgo_loss


value_rescale
================
Please refer to ``ding/rl_utils/value_rescale`` for more details.

value_transform
~~~~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.value_rescale.value_transform

value_inv_transform
~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.value_rescale.value_inv_transform

symlog
~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.value_rescale.symlog

inv_symlog
~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.value_rescale.inv_symlog

vtrace
========
Please refer to ``ding/rl_utils/vtrace`` for more details.

vtrace_nstep_return
~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.vtrace.vtrace_nstep_return

vtrace_advantage
~~~~~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.vtrace.vtrace_advantage

vtrace_data
~~~~~~~~~~~~~
.. autoclass:: ding.rl_utils.vtrace.vtrace_data

vtrace_loss
~~~~~~~~~~~~~
.. autoclass:: ding.rl_utils.vtrace.vtrace_loss

vtrace_error_discrete_action
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.vtrace.vtrace_error_discrete_action

vtrace_error_continuous_action
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: ding.rl_utils.vtrace.vtrace_error_continuous_action
