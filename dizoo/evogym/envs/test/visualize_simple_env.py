import gym
from evogym import sample_robot
from gym.wrappers import Monitor

# import envs from the envs folder and register them
import evogym.envs
from dizoo.evogym.envs.viewer import DingEvoViewer
from evogym.sim import EvoSim

if __name__ == '__main__':
    gym.logger.set_level(gym.logger.DEBUG)
    # create a random robot
    body, connections = sample_robot((5, 5))

    # make the SimpleWalkingEnv using gym.make and with the robot information
    #env = EvoGymEnv(EasyDict({'env_id': 'Walker-v0', 'robot': 'speed_bot', 'robot_dir': '../'}))
    #env.enable_save_replay('video')

    env = gym.make('Walker-v0', body=body)
    env.default_viewer = DingEvoViewer(EvoSim(env.world))
    env = Monitor(env, './video', force=True)
    env.__class__.render = env.default_viewer.render
    env.metadata['render.modes'] = 'rgb_array'

    env.reset()
    # step the environment for 200 iterations
    for i in range(100):
        action = env.action_space.sample()
        ob, reward, done, info = env.step(action)
        x = env.render()
        if done:
            env.reset()
    env.close()
