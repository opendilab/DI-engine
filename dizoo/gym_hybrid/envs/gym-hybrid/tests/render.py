import time
import gym
import gym_hybrid

if __name__ == '__main__':
    env = gym.make('Sliding-v0')
    env.reset()

    done = False
    while not done:
        _, _, done, _ = env.step(env.action_space.sample())
        env.render()
        time.sleep(0.1)

    time.sleep(1)
    env.close()
