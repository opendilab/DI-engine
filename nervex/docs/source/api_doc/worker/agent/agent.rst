worker.agent
===================

base_agent
-----------------

BaseAgent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.worker.agent.base_agent.BaseAgent
    :members: __init__, forward, mode, state_dict, load_state_dict, reset

AgentAggregator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.worker.agent.base_agent.AgentAggregator
    :members: __init__, __getattr__


agent_plugin
----------------------

IAgentPlugin
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.worker.agent.agent_plugin.IAgentPlugin
    :members: register


IAgentStatefulPlugin
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.worker.agent.agent_plugin.IAgentStatefulPlugin
    :members: __init__, reset

GradHelper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.worker.agent.agent_plugin.GradHelper
    :members: register

HiddenStateHelper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.worker.agent.agent_plugin.HiddenStateHelper
    :members: register

ArgmaxSampleHelper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.worker.agent.agent_plugin.ArgmaxSampleHelper
    :members: register

MultinomialSampleHelper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.worker.agent.agent_plugin.MultinomialSampleHelper
    :members: register

EpsGreedySampleHelper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.worker.agent.agent_plugin.EpsGreedySampleHelper
    :members: register

TargetNetworkHelper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.worker.agent.agent_plugin.TargetNetworkHelper
    :members: register, update, reset

TeacherNetworkHelper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.worker.agent.agent_plugin.TeacherNetworkHelper
    :members: register


register_plugin
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: nervex.worker.agent.agent_plugin.register_plugin


add_plugin
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: nervex.worker.agent.agent_plugin.add_plugin



agent_template
-----------------

create_dqn_learner_agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: nervex.worker.agent.agent_template.create_dqn_learner_agent


create_drqn_learner_agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: nervex.worker.agent.agent_template.create_drqn_learner_agent


create_ac_learner_agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: nervex.worker.agent.agent_template.create_ac_learner_agent


create_dqn_actor_agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: nervex.worker.agent.agent_template.create_dqn_actor_agent

create_drqn_actor_agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: nervex.worker.agent.agent_template.create_drqn_actor_agent

create_ac_actor_agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: nervex.worker.agent.agent_template.create_ac_actor_agent

create_dqn_evaluator_agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: nervex.worker.agent.agent_template.create_dqn_evaluator_agent

create_drqn_evaluator_agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: nervex.worker.agent.agent_template.create_drqn_evaluator_agent

create_ac_evaluator_agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: nervex.worker.agent.agent_template.create_ac_evaluator_agent
