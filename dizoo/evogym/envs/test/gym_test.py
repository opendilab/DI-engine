import gym
import evogym.envs
from evogym import sample_robot


if __name__ == '__main__':

    body, connections = sample_robot((5,5))
    env = gym.make('Walker-v0', body=body)
    env.reset()

    while True:
        action = env.action_space.sample()-1
        ob, reward, done, info = env.step(action)
        env.render()

        if done:
            env.reset()

    env.close()
