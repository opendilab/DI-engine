from typing import Any, List, Union, Optional
import gym
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.torch_utils import to_ndarray, to_list
from ding.utils import ENV_REGISTRY

@ENV_REGISTRY.register('mountain_car')
class MountainCar(BaseEnv):

    def __init__(self, cfg: dict = {}) -> None:
        self._cfg = cfg
        self._init_flag = False
        self._replay_path = None

        # Following specifications from https://is.gd/29S0dt
        self._observation_space = gym.spaces.Box(
            low=np.array([-1.2, -0.07]),
            high=np.array([0.6, 0.07]),
            shape=(2, ),
            dtype=np.float32
        )
        self._action_space = gym.spaces.Discrete(3)
        self._reward_space = gym.spaces.Box(low=-1, high=0.0, shape=(1, ), dtype=np.float32)

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def reset(self) -> np.ndarray:
        # Instantiate environment if not already done so
        if not self._init_flag:
            self._env = gym.make('MountainCar-v0')
        self._init_flag = True

        # Check if we have a valid replay path and save replay video accordingly
        if self._replay_path is not None:
            self._env = gym.wrappers.RecordVideo(
                self._env,
                video_folder=self._replay_path,
                episode_trigger=lambda episode_id: True,
                name_prefix='rl-video-{}'.format(id(self))
            )

        # Set the seeds for randomization.
        if hasattr(self, '_seed') and hasattr(self, '_dynamic_seed') and self._dynamic_seed:
            np_seed = 100 * np.random.randint(1, 1000)
            self._env.seed(self._seed + np_seed)
            self._action_space.seed(self._seed + np_seed)
        elif hasattr(self, '_seed'):
            self._env.seed(self._seed)
            self._action_space.seed(self._seed)
        
        # Get first observation from original environment
        obs = self._env.reset()

        # Convert to numpy array as output
        obs = to_ndarray(obs).astype(np.float32)

        # Init final reward : cumulative sum of the real rewards obtained by a whole episode, 
        # used to evaluate the agent Performance on this environment, not used for training.
        self._final_eval_reward = 0.
        return obs

    def step(self, action: np.ndarray) -> BaseEnvTimestep:

        # Making sure that input action is of numpy ndarray
        assert isinstance(action, np.ndarray), type(action)
        
        # Take a step of faith into the unknown!
        # c.f: https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py
        # obs : tuple of (position, velocity)
        obs, rew, done, info = self._env.step(action)

        # Cummulate reward
        self._final_eval_reward += rew
        
        # Save final cummulative reward when done.
        if done:
            info['final_eval_reward'] = self._final_eval_reward
        
        # Making sure we conform to di-engine conventions
        obs = to_ndarray(obs)                
        rew = to_ndarray([rew]).astype(np.float32)  

        return BaseEnvTimestep(obs, rew, done, info)

    def close(self) -> None:
        # If init flag is False, then reset() was never run, no point closing.
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    def enable_save_replay(self, replay_path: Optional[str] = None) -> None:
        if replay_path is None:
            replay_path = './video'
        self._replay_path = replay_path

    def random_action(self) -> np.ndarray:
        random_action = self._action_space.sample()
        random_action = to_ndarray([random_action], dtype=np.int64)
        return random_action

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space

    def __repr__(self) -> str:
        return "DI-engine Mountain Car Env"

if __name__ == '__main__':
    
    env = MountainCar()
    env.seed(314, dynamic_seed=False)
    assert env._seed == 314
    obs = env.reset()
    assert obs.shape == (2, )
    for _ in range(5):
        env.reset()
        np.random.seed(314)
        print('=' * 60)
        for i in range(10):
            # Both ``env.random_action()``, and utilizing ``np.random`` as well as action space,
            # can generate legal random action.
            if i < 5:
                random_action = np.array([env.action_space.sample()])
            else:
                random_action = env.random_action()
            timestep = env.step(random_action)
            print(timestep)
            assert isinstance(timestep.obs, np.ndarray)
            assert isinstance(timestep.done, bool)
            assert timestep.obs.shape == (4, )
            assert timestep.reward.shape == (1, )
            assert timestep.reward >= env.reward_space.low
            assert timestep.reward <= env.reward_space.high
    print(env.observation_space, env.action_space, env.reward_space)
    env.close()