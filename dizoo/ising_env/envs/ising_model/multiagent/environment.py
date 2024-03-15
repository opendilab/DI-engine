import gym
from gym import spaces
import numpy as np


class IsingMultiAgentEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(
        self,
        world,
        reset_callback=None,
        reward_callback=None,
        observation_callback=None,
        info_callback=None,
        done_callback=None
    ):

        self.world = world
        self.agents = self.world.policy_agents
        # number of controllable agents
        self.n = len(world.policy_agents)
        assert self.n == len(world.agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        # environment parameters
        self.discrete_action_space = True

        # if true, every agent has the same reward
        self.shared_reward = False
        self.time = 0

        # configure spaces
        self.action_space = []
        self.observation_space = []

        if self.discrete_action_space:
            self.action_space.append(spaces.Discrete(self.world.dim_spin))
        else:
            raise NotImplementedError("only discrete action is allowed")

        # observation space, called self-defined scenario.observation
        # define the size of the observation here
        # use the global state + mask
        # self.observation_space.append(spaces.MultiBinary(self.n * 2))
        self.observation_space.append(spaces.MultiBinary(4 * self.world.agent_view_sight))

    def _step(self, action_n):
        "descend from gym.env, env.step() actually calls this step"
        obs_n = []
        reward_n = []
        done_n = []
        self.agents = self.world.policy_agents

        display_action = action_n.reshape((int(np.sqrt(self.n)), -1))

        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent)

        # update the agent's new state and global state
        # world(in core.py) is a part of env, created though examples' make_world
        self.world.step()

        # observation, reward, done function are implemented in different examples
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))

        # all agents get total reward in cooperative case
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [reward] * self.n

        return obs_n, reward_n, done_n, self.world.order_param, \
            self.world.n_up, self.world.n_down

    def _reset(self):
        # reset world, init agent state and global state
        self.reset_callback(self.world)
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        return obs_n

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        # call the scenario's observation here
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        # call the scenario's done here
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        # call the scenario's reward here
        return self.reward_callback(agent, self.world)

    # set env action for a particular agent
    def _set_action(self, action, agent):
        agent.action.a = 0 if action <= 0 else 1
        assert len(action) == 1, "action dimenion error!"
