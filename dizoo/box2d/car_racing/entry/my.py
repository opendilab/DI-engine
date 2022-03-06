import gym
env = gym.make('CarRacing-v0')
obs = env.reset()
for _ in range(200):
    env.render()
    env.step(env.action_space.sample())
print(obs.shape)