from pettingzoo.mpe import simple_spread_v2

# Create a simple_spread_v2 environment
env = simple_spread_v2.parallel_env(N=3, local_ratio=0.5, max_cycles=25, continuous_actions=False)

# Reset the environment and get the initial observation
obs = env.reset()

# Print the number of agents and action spaces
print("=" * 40)
print("Number of agents:", env.num_agents)
print("Observation spaces: ", env.observation_spaces)
print("Action spaces: ", env.action_spaces)

# Print the shape of the Observation/action space of all possible_agents
for agent in env.possible_agents:
    print(f"Observation space shape of {agent}:", env.action_space(agent).shape)
    print(f"Action space shape of {agent}:", env.action_space(agent).shape)

print("=" * 40)
# Run 10 steps
for i in range(10):
    # Randomly choose an action for each agent and execute
    actions = {}
    for agent in env.possible_agents:
        actions.update({agent: env.action_space(agent).sample()})
    obs, rew, done, trunc, info = env.step(actions)
    print("Step:", i, "Reward:", rew)

# Close the environment
env.close()
