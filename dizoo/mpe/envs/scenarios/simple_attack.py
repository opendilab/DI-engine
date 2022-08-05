import numpy as np
from dizoo.mpe.envs.core import World, Agent, Landmark
from dizoo.mpe.envs.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self, args):
        world = World()
        # set any world properties first
        #world.dim_c = 2
        num_good_agents = args.num_good_agents#1
        num_adversaries = args.num_adversaries#3
        num_agents = num_adversaries + num_good_agents
        num_landmarks = args.num_landmarks#3
        assert num_landmarks==num_agents, ("should use the same number!")
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False # adversary first
            agent.size = 0.075 #if agent.adversary else 0.05
            agent.accel = 3.0 #if agent.adversary else 4.0
            #agent.accel = 20.0 if agent.adversary else 25.0
            agent.max_speed = 1.0 #if agent.adversary else 1.3
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False
        # make initial conditions
        self.reset_world(world)
        self.agent_failed = False
        return world

    def reset_world(self, world):
        # random properties for agents
        world.assign_agent_colors()
        # random properties for landmarks
        world.assign_landmark_colors()
        # random properties for landmarks
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            #agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = 0.8 * np.random.uniform(-1, +1, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)
                world.agents[i].goal = landmark


    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]


    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)
        return main_reward

    # agents are penalized for exiting the screen, so that they can be caught by the adversaries
    def bound(x):
        if x < 0.9:
            return 0
        if x < 1.0:
            return (x - 0.9) * 10
        return min(np.exp(2 * x - 2), 10)

    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0

        # goal distance
        goal_dist = np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal.state.p_pos)))
        rew -= goal_dist
        if goal_dist < agent.goal.size:
            rew += 0.5
      
        adversaries = self.adversaries(world)       
        for adv in adversaries:
            adv_dist = np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))
            if adv_dist < 0.15: # adv distance
                rew -= 0.1
            if agent.collide: # collide
                if adv_dist < (agent.size + adv.size):
                    rew -= 0.5
                self.agent_failed = True

        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0

        # goal distance
        goal_dist = np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal.state.p_pos)))
        rew -= goal_dist
        if goal_dist < agent.goal.size:
            rew += 0.5

        # collide
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if agent.collide:
            for ag in agents:
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        rew -= 0.5
        
        for p in range(world.dim_p):
            x = abs(agent.state.p_pos[p])
            rew -= bound(x)

        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        # comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            if other is agent: continue
            #comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            other_vel.append(other.state.p_vel)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + other_vel)

    def info(self, agent, world):
        info = {}
        info['fail'] = self.agent_failed
        return info