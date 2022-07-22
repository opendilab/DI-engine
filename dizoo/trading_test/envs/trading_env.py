from cmath import inf
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY
import os
import pandas as pd
from copy import deepcopy
from ding.torch_utils import to_ndarray


def load_dataset(name, index_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, 'data', name + '.csv')
    df = pd.read_csv(path, parse_dates=True, index_col=index_name)
    return df

class Actions(Enum):
    Double_Sell = 0
    Sell = 1
    Hold = 2
    Buy = 3
    Double_Buy = 4


class Positions(Enum):
    Short = -1.
    Flat = 0.
    Long = 1.

def trans(position,  action):

    if action == Actions.Sell.value:
        
        if position == Positions.Long:
            return Positions.Flat, False
            
        if position == Positions.Flat:
            return Positions.Short, True

    if action == Actions.Buy.value:

        if position == Positions.Short:
            return Positions.Flat, False

        if position == Positions.Flat:
            return Positions.Long, True
    
    if action == Actions.Double_Sell.value and (position == Positions.Long or position == Positions.Flat):
        return Positions.Short, True

    if action == Actions.Double_Buy.value and (position == Positions.Short or position == Positions.Flat):
        return Positions.Long, True

    return position, False


@ENV_REGISTRY.register('base_trading')
class TradingEnv(BaseEnv):

    def __init__(self, cfg):

        self._cfg = cfg
        self.cnt = 0 # associate the frequence that update profit.png
        STOCKS_GOOGL = load_dataset('STOCKS_GOOGL', 'Date')
        self.raw_prices = deepcopy(STOCKS_GOOGL).loc[:, 'Close'].to_numpy()
        self.df = deepcopy(STOCKS_GOOGL).apply(lambda x: (x-x.mean())/ x.std(), axis=0) # normalize
        

        self.window_size = cfg.window_size
        self.prices = None
        self.signal_features = None
        self.shape = (cfg.window_size, 3)


        # episode
        self._start_tick = None
        self._end_tick = None
        self._done = None
        self._current_tick = None
        self._last_trade_tick = None
        self._position = None
        self._position_history = None
        self._total_reward = None
        

        self._env_id = cfg.env_id
        self._action_space = spaces.Discrete(len(Actions))
        self._observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float64)
        self._reward_space = gym.spaces.Box(
                -inf, inf, shape=(1, ), dtype=np.float32
            )


    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)
        self.np_random, seed = seeding.np_random(seed)



    def reset(self, start_idx = None):
        self.cnt += 1
        self.prices, self.signal_features = self._process_data(start_idx)
        self._done = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._position = Positions.Flat
        self._position_history = [self._position]
        self._profit_history = [1.]
        self._total_reward = 0.



        #print("#############",self._start_tick, self._end_tick)
        return self._get_observation()


    def step(self, action):
        action = int(action[0])
        
        self._done = False
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._done = True

        step_reward = self._calculate_reward(action)
        self._total_reward += step_reward

        
        self._position, trade = trans( self._position, action)

        if trade:
            self._last_trade_tick = self._current_tick

        self._position_history.append(self._position)
        self._profit_history.append(float(np.exp(self._total_reward)))
        observation = self._get_observation()
        info = dict(
            total_reward = self._total_reward,
            position = self._position.value,
            
        )

        if self._done:
            if self.cnt % 10 == 0:
                self.render()
            info['max_possible_profit'] = self.max_possible_profit()
            info['final_eval_reward'] = self._total_reward


            
        
        return BaseEnvTimestep(observation, step_reward, self._done, info)


    def _get_observation(self):
        obs = to_ndarray(self.signal_features[(self._current_tick-self.window_size+1):self._current_tick+1]).reshape(-1).astype(np.float32)
        obs = np.hstack([obs, to_ndarray([self._position.value])]).astype(np.float32)

        return obs



    def render(self, save_path = '/home/PJLAB/chenyun/test_pic/'):
        plt.clf()
        plt.plot(self._profit_history)
        plt.savefig(save_path + str(self._env_id) + "profit.png")


        plt.clf()
        window_ticks = np.arange(len(self._position_history))
        eps_price = self.raw_prices[self._start_tick:self._end_tick+1]
        plt.plot(eps_price)
        

        short_ticks = []
        long_ticks = []
        flat_ticks = []
        for i, tick in enumerate(window_ticks):
            if self._position_history[i] == Positions.Short:
                short_ticks.append(tick)
            elif self._position_history[i] == Positions.Long:
                long_ticks.append(tick)
            else:
                flat_ticks.append(tick)

        plt.plot(short_ticks, eps_price[short_ticks], 'ro')
        plt.plot(long_ticks, eps_price[long_ticks], 'go')
        plt.plot(flat_ticks, eps_price[flat_ticks], 'bo')
        plt.savefig(save_path + str(self._env_id) + 'price.png')

        
    def close(self):
        plt.close()


    def _process_data(self):
        raise NotImplementedError


    def _calculate_reward(self, action):
        raise NotImplementedError


    def _update_profit(self, action):
        raise NotImplementedError


    def max_possible_profit(self):  # trade fees are ignored
        raise NotImplementedError


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
        return "DI-engine Trading Env"
