import numpy as np


class IsingEntityState(object):

    def __init__(self):
        self.id = None
        self.p_pos = None


class IsingAgentState(IsingEntityState):

    def __init__(self):
        super(IsingAgentState, self).__init__()
        # up or down
        self.spin = None


class IsingAction(object):

    def __init__(self):
        # action
        self.a = None


# properties and state of physical world entity
class IsingEntity(object):

    def __init__(self):
        # name
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # color
        self.color = None
        # state: position and spin
        self.state = IsingEntityState()


class IsingAgent(IsingEntity):

    def __init__(self, view_sight=1):
        super(IsingAgent, self).__init__()
        # agents are movable by default
        self.movable = False
        # -1: observe the whole state, 0: itself, 1: neighbour of 1 unit
        self.view_sight = view_sight
        self.spin_mask = None  # the mask for who is neighbours
        # state
        self.state = IsingAgentState()
        self.state.spin_range = [0, 1]
        # action
        self.action = IsingAction()
        self.action.a_range = [0, 1]
        # script behavior to execute
        self.action_callback = None


# multi-agent world
class IsingWorld(object):

    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.n_agents = 1
        self.agent_view_sight = 1
        # position dimensionality
        self.dim_pos = 2
        # state dimension
        self.dim_spin = 2
        # color dimensionality
        self.dim_color = 3
        # world size
        self.shape_size = 1
        # ising specific
        self.global_state = None  # log all spins
        self.moment = 1
        self.field = None  # external magnetic field
        self.temperature = .1  # Temperature (in units of energy)
        self.interaction = 1  # Interaction (ferromagnetic if positive,
        # antiferromagnetic if negative)
        self.order_param = 1.0
        self.order_param_delta = 0.01  # log the change of order parameter for "done"
        self.n_up = 0
        self.n_down = 0

    # return all entities in the world
    @property
    def entities(self):
        return self.agents

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts, no use for now
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    def step(self):

        # set actions for scripted agents, no use for now
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)

        # update agent state, and to the global_state
        for agent in self.agents:
            self.update_agent_state(agent)
            self.global_state[agent.state.p_pos] = agent.state.spin

        # update the world's order parameters
        self.n_up = np.count_nonzero(self.global_state.flatten())
        self.n_down = self.n_agents - self.n_up
        order_param_old = self.order_param
        self.order_param = abs(self.n_up - self.n_down) / (self.n_agents + 0.0)
        # self.order_param_delta = (self.order_param - order_param_old) /
        # order_param_old

    def update_agent_state(self, agent):
        if agent.action.a == 0:
            # agent.state.spin = agent.state.spin
            agent.state.spin = 0
        else:
            # print(agent.name + " change spin")
            # agent.state.spin = 1.0 - agent.state.spin
            agent.state.spin = 1
