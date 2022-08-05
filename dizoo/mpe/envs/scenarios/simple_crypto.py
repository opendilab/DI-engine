"""
Scenario:
1 speaker, 2 listeners (one of which is an adversary). Good agents rewarded for proximity to goal, and distance from
adversary to goal. Adversary is rewarded for its distance to the goal.
"""


import numpy as np
from onpolicy.envs.mpe.core import World, Agent, Landmark
from onpolicy.envs.mpe.scenario import BaseScenario
import random


class CryptoAgent(Agent):
    def __init__(self):
        super(CryptoAgent, self).__init__()
        self.key = None

class Scenario(BaseScenario):

    def make_world(self,args):
        world = World()
        # set any world properties first
        num_agents = args.num_agents#3
        num_adversaries = 1
        num_landmarks = args.num_landmarks#2
        world.dim_c = 4
        # add agents
        world.agents = [CryptoAgent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.adversary = True if i < num_adversaries else False
            agent.speaker = True if i == 2 else False
            agent.movable = False
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world


    def reset_world(self, world):
        # random properties for agents
        for agent in world.agents:
            agent.color = np.array([0.25, 0.25, 0.25])
            if agent.adversary:
                agent.color = np.array([0.75, 0.25, 0.25])
            agent.key = None
        # random properties for landmarks
        color_list = [np.zeros(world.dim_c) for i in world.landmarks]
        for i, color in enumerate(color_list):
            color[i] += 1
        for color, landmark in zip(color_list, world.landmarks):
            landmark.color = color
        # set goal landmark
        goal = np.random.choice(world.landmarks)
        world.agents[1].color = goal.color
        world.agents[2].key = np.random.choice(world.landmarks).color

        for agent in world.agents:
            agent.goal_a = goal

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)


    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        return (agent.state.c, agent.goal_a.color)

    # return all agents that are not adversaries
    def good_listeners(self, world):
        return [agent for agent in world.agents if not agent.adversary and not agent.speaker]

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        return self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)

    def agent_reward(self, agent, world):
        # Agents rewarded if Bob can reconstruct message, but adversary (Eve) cannot
        good_listeners = self.good_listeners(world)
        adversaries = self.adversaries(world)
        good_rew = 0
        adv_rew = 0
        for a in good_listeners:
            if (a.state.c == np.zeros(world.dim_c)).all():
                continue
            else:
                good_rew -= np.sum(np.square(a.state.c - agent.goal_a.color))
        for a in adversaries:
            if (a.state.c == np.zeros(world.dim_c)).all():
                continue
            else:
                adv_l1 = np.sum(np.square(a.state.c - agent.goal_a.color))
                adv_rew += adv_l1
        return adv_rew + good_rew

    def adversary_reward(self, agent, world):
        # Adversary (Eve) is rewarded if it can reconstruct original goal
        rew = 0
        if not (agent.state.c == np.zeros(world.dim_c)).all():
            rew -= np.sum(np.square(agent.state.c - agent.goal_a.color))
        return rew


    def observation(self, agent, world):
        # goal color
        goal_color = np.zeros(world.dim_color)
        if agent.goal_a is not None:
            goal_color = agent.goal_a.color

        #print('goal color in obs is {}'.format(goal_color))

        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        for other in world.agents:
            if other is agent or (other.state.c is None) or not other.speaker: continue
            comm.append(other.state.c)

        confer = np.array([0])

        if world.agents[2].key is None:
            confer = np.array([1])
            key = np.zeros(world.dim_c)
            goal_color = np.zeros(world.dim_c)
        else:
            key = world.agents[2].key

        prnt = False#True  if train use False
        # speaker
        if agent.speaker:
            if prnt:
                print('speaker')
                print(agent.state.c)
                print(np.concatenate([goal_color] + [key] + [confer] + [np.random.randn(1)]))
            return np.concatenate([goal_color] + [key])
        # listener
        if not agent.speaker and not agent.adversary:
            if prnt:
                print('listener')
                print(agent.state.c)
                print(np.concatenate([key] + comm + [confer]))
            return np.concatenate([key] + comm)
        if not agent.speaker and agent.adversary:
            if prnt:
                print('adversary')
                print(agent.state.c)
                print(np.concatenate(comm + [confer]))
            return np.concatenate(comm)
