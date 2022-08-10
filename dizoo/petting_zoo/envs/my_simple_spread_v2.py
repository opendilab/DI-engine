from pettingzoo.utils.conversions import parallel_wrapper_fn

from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.mpe.scenarios.simple_spread import Scenario


class raw_env(SimpleEnv):
    def __init__(self, N=3, local_ratio=0.5, max_cycles=25, continuous_actions=False):
        assert 0. <= local_ratio <= 1., "local_ratio is a proportion. Must be between 0 and 1."
        scenario = Scenario()
        world = scenario.make_world(N)
        super().__init__(scenario, world, max_cycles, continuous_actions, local_ratio)
        self.metadata['name'] = "simple_spread_v2"

    def _execute_world_step(self):
        # set action for each agent
        for i, agent in enumerate(self.world.agents):
            action = self.current_actions[i]
            scenario_action = []
            if agent.movable:
                mdim = self.world.dim_p * 2 + 1
                if self.continuous_actions:
                    scenario_action.append(action[0:mdim])
                    action = action[mdim:]
                else:
                    scenario_action.append(action % mdim)
                    action //= mdim
            if not agent.silent:
                scenario_action.append(action)
            self._set_action(scenario_action, agent, self.action_spaces[agent.name])

        self.world.step()

        global_reward = 0.
        if self.local_ratio is not None:
            global_reward = float(self.scenario.global_reward(self.world))

        for agent in self.world.agents:
            agent_reward = float(self.scenario.reward(agent, self.world))
            if self.local_ratio is not None:
                # we changed reward calc way to keep same with mpe
                # reward = global_reward * (1 - self.local_ratio) + agent_reward * self.local_ratio
                reward = global_reward + agent_reward
            else:
                reward = agent_reward

            self.rewards[agent.name] = reward


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)
