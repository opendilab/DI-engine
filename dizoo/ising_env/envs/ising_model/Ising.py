import numpy as np

from dizoo.ising_env.envs.ising_model.multiagent.core import IsingWorld, IsingAgent


class Scenario():

    def _calc_mask(self, agent, shape_size):
        # compute the neighbour mask for each agent
        if agent.view_sight == -1:
            # fully observed
            agent.spin_mask += 1
        elif agent.view_sight == 0:
            # observe itself
            agent.spin_mask[agent.state.id] = 1
        elif agent.view_sight > 0:
            # observe neighbours
            delta = list(range(-int(agent.view_sight), int(agent.view_sight) + 1, 1))
            delta.remove(0)  # agent itself is not counted as neighbour of itself
            for dt in delta:
                row = agent.state.p_pos[0]
                col = agent.state.p_pos[1]
                row_dt = row + dt
                col_dt = col + dt
                if row_dt in range(0, shape_size):
                    agent.spin_mask[agent.state.id + shape_size * dt] = 1
                if col_dt in range(0, shape_size):
                    agent.spin_mask[agent.state.id + dt] = 1

            # the graph is cyclic, most left and most right are neighbours
            if agent.state.p_pos[0] < agent.view_sight:
                tar = shape_size - (np.array(range(0, int(agent.view_sight - agent.state.p_pos[0]), 1)) + 1)
                tar = tar * shape_size + agent.state.p_pos[1]
                agent.spin_mask[tar] = [1] * len(tar)

            if agent.state.p_pos[1] < agent.view_sight:
                tar = shape_size - (np.array(range(0, int(agent.view_sight - agent.state.p_pos[1]), 1)) + 1)
                tar = agent.state.p_pos[0] * shape_size + tar
                agent.spin_mask[tar] = [1] * len(tar)

            if agent.state.p_pos[0] >= shape_size - agent.view_sight:
                tar = np.array(range(0, int(agent.view_sight - (shape_size - 1 - agent.state.p_pos[0])), 1))
                tar = tar * shape_size + agent.state.p_pos[1]
                agent.spin_mask[tar] = [1] * len(tar)

            if agent.state.p_pos[1] >= shape_size - agent.view_sight:
                tar = np.array(range(0, int(agent.view_sight - (shape_size - 1 - agent.state.p_pos[1])), 1))
                tar = agent.state.p_pos[0] * shape_size + tar
                agent.spin_mask[tar] = [1] * len(tar)

    def make_world(self, num_agents=100, agent_view=1):
        world = IsingWorld()
        world.agent_view_sight = agent_view
        world.dim_spin = 2
        world.dim_pos = 2
        world.n_agents = num_agents
        world.shape_size = int(np.ceil(np.power(num_agents, 1.0 / world.dim_pos)))
        world.global_state = np.zeros((world.shape_size, ) * world.dim_pos)
        # assume 0 external magnetic field
        world.field = np.zeros((world.shape_size, ) * world.dim_pos)

        world.agents = [IsingAgent(view_sight=world.agent_view_sight) for i in range(num_agents)]

        # make initial conditions
        self.reset_world(world)

        return world

    def reset_world(self, world):

        world_mat = np.array(
                          range(np.power(world.shape_size, world.dim_pos))). \
                          reshape((world.shape_size,) * world.dim_pos)
        # init agent state and global state
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.color = np.array([0.35, 0.35, 0.85])
            agent.state.id = i
            agent.state.p_pos = np.where(world_mat == i)
            agent.state.spin = np.random.choice(world.dim_spin)
            agent.spin_mask = np.zeros(world.n_agents)

            assert world.dim_pos == 2, "cyclic neighbour only support 2D now"
            self._calc_mask(agent, world.shape_size)
            world.global_state[agent.state.p_pos] = agent.state.spin

        n_ups = np.count_nonzero(world.global_state.flatten())
        n_downs = world.n_agents - n_ups
        world.order_param = abs(n_ups - n_downs) / (world.n_agents + 0.0)

    def reward(self, agent, world):
        # turn the state into -1/1 for easy computing
        world.global_state[np.where(world.global_state == 0)] = -1

        mask_display = agent.spin_mask.reshape((int(np.sqrt(world.n_agents)), -1))

        local_reward = - 0.5 * world.global_state[agent.state.p_pos] \
            * np.sum(world.global_state.flatten() * agent.spin_mask)

        world.global_state[np.where(world.global_state == -1)] = 0
        return -local_reward

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        # agent state is updated in the world.step() function already
        # update the changes of the world

        # return the neighbour state
        return world.global_state.flatten()[np.where(agent.spin_mask == 1)]

    def done(self, agent, world):
        if world.order_param == 1.0:
            return True
        return False
