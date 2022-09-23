import gym
from evogym import sample_robot
from gym.wrappers import Monitor

# import envs from the envs folder and register them
import evogym.envs
from ..viewer import DingEvoViewer
from evogym.sim import EvoSim

if __name__ == '__main__':
    gym.logger.set_level(gym.logger.DEBUG)
    # create a random robot
    body, connections = sample_robot((5, 5))

    # make the SimpleWalkingEnv using gym.make and with the robot information
    env = gym.make('Walker-v0', body=body)
    #env = gym.make('Pusher-v0', body=body)
    env.metadata['render.modes'] = 'rgb_array'
    env._default_viewer = DingEvoViewer(EvoSim(env.world))
    env.metadata['render.modes'] = 'rgb_array'  # make render mode compatible with gym
    env = Monitor(env, './video', force=True)
    env.reset()
    # step the environment for 200 iterations
    for i in range(200):
        action = env.action_space.sample()
        ob, reward, done, info = env.step(action)
        env.render()
        if done:
            env.reset()
    env.close()
