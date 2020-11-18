from .agent_plugin import add_plugin, IAgentStatefulPlugin, IAgentStatelessPlugin
from .base_agent import BaseAgent, AgentAggregator
from .agent_template import create_dqn_learner_agent, create_dqn_actor_agent, create_dqn_evaluator_agent,\
    create_drqn_learner_agent, create_drqn_actor_agent, create_drqn_evaluator_agent,\
    create_ac_learner_agent, create_ac_actor_agent, create_ac_evaluator_agent
