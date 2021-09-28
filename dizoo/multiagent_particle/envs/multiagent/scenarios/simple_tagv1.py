import numpy as np
from dizoo.multiagent_particle.envs.multiagent.core import World, Agent, Landmark
from dizoo.multiagent_particle.envs.multiagent.scenario import BaseScenario
from easydict import EasyDict


def random_action(*args):
    # Todo 5 should be action dim
    return np.random.choice(5)


class Scenario(BaseScenario):

    def make_world(self, num_agents=4, num_landmarks=2, num_good_agents=1, cfg=None):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_adversaries = num_agents - num_good_agents
        cfg = EasyDict(cfg)
        self.num_catch = cfg.num_catch
        self.reward_right_catch = cfg.reward_right_catch
        self.reward_wrong_catch = cfg.reward_wrong_catch
        self.collision_ratio = cfg.collision_ratio
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            # agent.size = 0.075 if agent.adversary else 0.05
            agent.size = 0.15 if agent.adversary else 0.1
            agent.accel = 3.0 if agent.adversary else 4.0
            # agent.accel = 20.0 if agent.adversary else 25.0
            agent.max_speed = 1.0 if agent.adversary else 1.3
            # agent.action_callback = None if agent.adversary else random_action
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
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.85, 0.35]) if not agent.adversary else np.array([0.85, 0.35, 0.35])
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
                landmark.state.p_vel = np.zeros(world.dim_p)

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
        return True if dist < dist_min * self.collision_ratio else False

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

    def agent_reward(self, agent, world):
        rew = 0
        return rew

    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        shape = False
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        if shape:  # reward can optionally be shaped (decreased reward for increased distance from agents)
            for adv in adversaries:
                rew -= 0.1 * min([np.sqrt(np.sum(np.square(a.state.p_pos - adv.state.p_pos))) for a in agents])
        if agent.collide:
            for ag in agents:
                catch = 0
                for adv in adversaries:
                    if self.is_collision(ag, adv):
                        catch += 1
                if catch >= self.num_catch:
                    rew += self.reward_right_catch * catch
                elif catch > 0:
                    rew += self.reward_wrong_catch
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        # used in global state
        prey_pos = []
        prey_vel = []
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            # if not other.adversary:
            other_vel.append(other.state.p_vel)
            if not other.adversary:
                prey_pos.append(other.state.p_pos)
                prey_vel.append(other.state.p_vel)
        return np.concatenate(
            [agent.state.p_pos] + [agent.state.p_vel] + other_pos + other_vel + prey_pos + prey_vel + entity_pos
        )
