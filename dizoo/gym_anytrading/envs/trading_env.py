from cmath import inf
from typing import Any, List
from easydict import EasyDict
from abc import abstractmethod
from gym import spaces
from gym.utils import seeding
from enum import Enum

import os
import gym
import copy
import pandas as pd
import numpy as np

from ding.envs import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY
from ding.torch_utils import to_ndarray


def load_dataset(name, index_name):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, 'data', name + '.csv')
    assert os.path.exists(
        path
    ), "You need to put the stock data under the \'DI-engine/dizoo/gym_anytrading/envs/data\' folder.\n \
        if using StocksEnv, you can download Google stocks data at \
        https://github.com/AminHP/gym-anytrading/blob/master/gym_anytrading/datasets/data/STOCKS_GOOGL.csv"

    df = pd.read_csv(path, parse_dates=True, index_col=index_name)
    return df


class Actions(int, Enum):
    DOUBLE_SELL = 0
    SELL = 1
    HOLD = 2
    BUY = 3
    DOUBLE_BUY = 4


class Positions(int, Enum):
    SHORT = -1.
    FLAT = 0.
    LONG = 1.


def transform(position: Positions, action: int) -> Any:
    '''
    Overview:
        used by env.tep().
        This func is used to transform the env's position from
        the input (position, action) pair according to the status machine.
    Arguments:
        - position(Positions) : Long, Short or Flat
        - action(int) : Doulbe_Sell, Sell, Hold, Buy, Double_Buy
    Returns:
        - next_position(Positions) : the position after transformation.
    '''
    if action == Actions.SELL:

        if position == Positions.LONG:
            return Positions.FLAT, False

        if position == Positions.FLAT:
            return Positions.SHORT, True

    if action == Actions.BUY:

        if position == Positions.SHORT:
            return Positions.FLAT, False

        if position == Positions.FLAT:
            return Positions.LONG, True

    if action == Actions.DOUBLE_SELL and (position == Positions.LONG or position == Positions.FLAT):
        return Positions.SHORT, True

    if action == Actions.DOUBLE_BUY and (position == Positions.SHORT or position == Positions.FLAT):
        return Positions.LONG, True

    return position, False


@ENV_REGISTRY.register('base_trading')
class TradingEnv(BaseEnv):

    def __init__(self, cfg: EasyDict) -> None:

        self._cfg = cfg
        self._env_id = cfg.env_id
        #======== param to plot =========
        self.cnt = 0

        if 'plot_freq' not in self._cfg:
            self.plot_freq = 10
        else:
            self.plot_freq = self._cfg.plot_freq
        if 'save_path' not in self._cfg:
            self.save_path = './'
        else:
            self.save_path = self._cfg.save_path
        #================================

        self.train_range = cfg.train_range
        self.test_range = cfg.test_range
        self.window_size = cfg.window_size
        self.prices = None
        self.signal_features = None
        self.feature_dim_len = None
        self.shape = (cfg.window_size, 3)

        #======== param about episode =========
        self._start_tick = 0
        self._end_tick = 0
        self._done = None
        self._current_tick = None
        self._last_trade_tick = None
        self._position = None
        self._position_history = None
        self._total_reward = None
        #======================================

        self._init_flag = True
        # init the following variables variable at first reset.
        self._action_space = None
        self._observation_space = None
        self._reward_space = None

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)
        self.np_random, seed = seeding.np_random(seed)

    def reset(self, start_idx: int = None) -> Any:
        self.cnt += 1
        self.prices, self.signal_features, self.feature_dim_len = self._process_data(start_idx)
        if self._init_flag:
            self.shape = (self.window_size, self.feature_dim_len)
            self._action_space = spaces.Discrete(len(Actions))
            self._observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float64)
            self._reward_space = gym.spaces.Box(-inf, inf, shape=(1, ), dtype=np.float32)
            self._init_flag = False
        self._done = False
        self._current_tick = self._start_tick
        self._last_trade_tick = self._current_tick - 1
        self._position = Positions.FLAT
        self._position_history = [self._position]
        self._profit_history = [1.]
        self._total_reward = 0.

        return self._get_observation()

    def random_action(self) -> Any:
        return np.array([self.action_space.sample()])

    def step(self, action: np.ndarray) -> BaseEnvTimestep:
        assert isinstance(action, np.ndarray), type(action)
        if action.shape == (1, ):
            action = action.item()  # 0-dim array

        self._done = False
        self._current_tick += 1

        if self._current_tick >= self._end_tick:
            self._done = True

        step_reward = self._calculate_reward(action)
        self._total_reward += step_reward

        self._position, trade = transform(self._position, action)

        if trade:
            self._last_trade_tick = self._current_tick

        self._position_history.append(self._position)
        self._profit_history.append(float(np.exp(self._total_reward)))
        observation = self._get_observation()
        info = dict(
            total_reward=self._total_reward,
            position=self._position.value,
        )

        if self._done:
            if self._env_id[-1] == 'e' and self.cnt % self.plot_freq == 0:
                self.render()
            info['max_possible_profit'] = np.log(self.max_possible_profit())
            info['final_eval_reward'] = self._total_reward

        step_reward = to_ndarray([step_reward]).astype(np.float32)
        return BaseEnvTimestep(observation, step_reward, self._done, info)

    def _get_observation(self) -> np.ndarray:
        obs = to_ndarray(self.signal_features[(self._current_tick - self.window_size + 1):self._current_tick + 1]
                         ).reshape(-1).astype(np.float32)

        tick = (self._current_tick - self._last_trade_tick) / self._cfg.eps_length
        obs = np.hstack([obs, to_ndarray([self._position.value]), to_ndarray([tick])]).astype(np.float32)
        return obs

    def render(self) -> None:
        import matplotlib.pyplot as plt
        plt.clf()
        plt.xlabel('trading days')
        plt.ylabel('profit')
        plt.plot(self._profit_history)
        plt.savefig(self.save_path + str(self._env_id) + "-profit.png")

        plt.clf()
        plt.xlabel('trading days')
        plt.ylabel('close price')
        window_ticks = np.arange(len(self._position_history))
        eps_price = self.raw_prices[self._start_tick:self._end_tick + 1]
        plt.plot(eps_price)

        short_ticks = []
        long_ticks = []
        flat_ticks = []
        for i, tick in enumerate(window_ticks):
            if self._position_history[i] == Positions.SHORT:
                short_ticks.append(tick)
            elif self._position_history[i] == Positions.LONG:
                long_ticks.append(tick)
            else:
                flat_ticks.append(tick)

        plt.plot(long_ticks, eps_price[long_ticks], 'g^', markersize=3, label="Long")
        plt.plot(flat_ticks, eps_price[flat_ticks], 'bo', markersize=3, label="Flat")
        plt.plot(short_ticks, eps_price[short_ticks], 'rv', markersize=3, label="Short")
        plt.legend(loc='upper left', bbox_to_anchor=(0.05, 0.95))
        plt.savefig(self.save_path + str(self._env_id) + '-price.png')

    def close(self):
        import matplotlib.pyplot as plt
        plt.close()

    # override
    def create_collector_env_cfg(cfg: dict) -> List[dict]:
        """
        Overview:
            Return a list of all of the environment from input config, used in env manager \
            (a series of vectorized env), and this method is mainly responsible for envs collecting data.
            In TradingEnv, this method will rename every env_id and generate different config.
        Arguments:
            - cfg (:obj:`dict`): Original input env config, which needs to be transformed into the type of creating \
                env instance actually and generated the corresponding number of configurations.
        Returns:
            - env_cfg_list (:obj:`List[dict]`): List of ``cfg`` including all the config collector envs.
        .. note::
            Elements(env config) in collector_env_cfg/evaluator_env_cfg can be different, such as server ip and port.
        """
        collector_env_num = cfg.pop('collector_env_num')
        collector_env_cfg = [copy.deepcopy(cfg) for _ in range(collector_env_num)]
        for i in range(collector_env_num):
            collector_env_cfg[i]['env_id'] += ('-' + str(i) + 'e')
        return collector_env_cfg

    # override
    def create_evaluator_env_cfg(cfg: dict) -> List[dict]:
        """
        Overview:
            Return a list of all of the environment from input config, used in env manager \
            (a series of vectorized env), and this method is mainly responsible for envs evaluating performance.
            In TradingEnv, this method will rename every env_id and generate different config.
        Arguments:
            - cfg (:obj:`dict`): Original input env config, which needs to be transformed into the type of creating \
                env instance actually and generated the corresponding number of configurations.
        Returns:
            - env_cfg_list (:obj:`List[dict]`): List of ``cfg`` including all the config evaluator envs.
        """
        evaluator_env_num = cfg.pop('evaluator_env_num')
        evaluator_env_cfg = [copy.deepcopy(cfg) for _ in range(evaluator_env_num)]
        for i in range(evaluator_env_num):
            evaluator_env_cfg[i]['env_id'] += ('-' + str(i) + 'e')
        return evaluator_env_cfg

    @abstractmethod
    def _process_data(self):
        raise NotImplementedError

    @abstractmethod
    def _calculate_reward(self, action):
        raise NotImplementedError

    @abstractmethod
    def max_possible_profit(self):
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
