agent
===================

agent
-----------------

BaseAgent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.agent.agent.BaseAgent
    :members: __init__, forward, mode, state_dict, load_state_dict, reset

Agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.agent.agent.Agent
    :members: __init__, __getattr__


agent_plugin
----------------------

IAgentPlugin
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.agent.agent_plugin.IAgentPlugin
    :members: register


IAgentStatefulPlugin
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.agent.agent_plugin.IAgentStatefulPlugin
    :members: __init__, reset

GradHelper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.agent.agent_plugin.GradHelper
    :members: register

HiddenStateHelper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.agent.agent_plugin.HiddenStateHelper
    :members: register

ArgmaxSampleHelper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.agent.agent_plugin.ArgmaxSampleHelper
    :members: register

MultinomialSampleHelper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.agent.agent_plugin.MultinomialSampleHelper
    :members: register

EpsGreedySampleHelper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.agent.agent_plugin.EpsGreedySampleHelper
    :members: register

TargetNetworkHelper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.agent.agent_plugin.TargetNetworkHelper
    :members: register, update, reset

TeacherNetworkHelper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: nervex.agent.agent_plugin.TeacherNetworkHelper
    :members: register


register_plugin
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: nervex.agent.agent_plugin.register_plugin


add_plugin
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: nervex.agent.agent_plugin.add_plugin
