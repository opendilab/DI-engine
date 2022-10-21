import gym
import gym_hybrid

if __name__ == '__main__':
    env = gym.make('Sliding-v0')
    env = gym.wrappers.Monitor(env, "./video", force=True)
    env.metadata["render.modes"] = ["human", "rgb_array"]
    env.reset()

    done = False
    while not done:
        _, _, done, _ = env.step(env.action_space.sample())

    env.close()
