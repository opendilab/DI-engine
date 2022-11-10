import time
import gym
import gym_hybrid

if __name__ == '__main__':
    env = gym.make('Sliding-v0')
    env.reset()

    ACTION_SPACE = env.action_space[0].n
    PARAMETERS_SPACE = env.action_space[1].shape[0]
    OBSERVATION_SPACE = env.observation_space.shape[0]

    done = False
    while not done:
        state, reward, done, info = env.step(env.action_space.sample())
        print(f'State: {state} Reward: {reward} Done: {done}')
        time.sleep(0.1)
