from pprint import pprint
from typing import Any
from copy import deepcopy
import numpy as np

from dizoo.gym_anytrading.envs.trading_env import TradingEnv, Actions, Positions, load_dataset
from ding.utils import ENV_REGISTRY
from ding.torch_utils import to_ndarray


@ENV_REGISTRY.register('stocks-v0')
class StocksEnv(TradingEnv):

    def __init__(self, cfg):

        super().__init__(cfg)

        # ====== load Google stocks data =======
        raw_data = load_dataset(self._cfg.stocks_data_filename, 'Date')
        self.raw_prices = raw_data.loc[:, 'Close'].to_numpy()
        EPS = 1e-10
        self.df = deepcopy(raw_data)
        if self.train_range == None or self.test_range == None:
            self.df = self.df.apply(lambda x: (x - x.mean()) / (x.std() + EPS), axis=0)
        else:
            boundary = int(len(self.df) * self.train_range)
            train_data = raw_data[:boundary].copy()
            boundary = int(len(raw_data) * (1 + self.test_range))
            test_data = raw_data[boundary:].copy()

            train_data = train_data.apply(lambda x: (x - x.mean()) / (x.std() + EPS), axis=0)
            test_data = test_data.apply(lambda x: (x - x.mean()) / (x.std() + EPS), axis=0)
            self.df.loc[train_data.index, train_data.columns] = train_data
            self.df.loc[test_data.index, test_data.columns] = test_data
        # ======================================

        # set cost
        self.trade_fee_bid_percent = 0.01  # unit
        self.trade_fee_ask_percent = 0.005  # unit

    # override
    def _process_data(self, start_idx: int = None) -> Any:
        '''
        Overview:
            used by env.reset(), process the raw data.
        Arguments:
            - start_idx (int): the start tick; if None, then randomly select.
        Returns:
            - prices: the close.
            - signal_features: feature map
            - feature_dim_len: the dimension length of selected feature
        '''

        # ====== build feature map ========
        all_feature_name = ['Close', 'Open', 'High', 'Low', 'Adj Close', 'Volume']
        all_feature = {k: self.df.loc[:, k].to_numpy() for k in all_feature_name}
        # add feature "Diff"
        prices = self.df.loc[:, 'Close'].to_numpy()
        diff = np.insert(np.diff(prices), 0, 0)
        all_feature_name.append('Diff')
        all_feature['Diff'] = diff
        # =================================

        # you can select features you want
        selected_feature_name = ['Close', 'Diff', 'Volume']
        selected_feature = np.column_stack([all_feature[k] for k in selected_feature_name])
        feature_dim_len = len(selected_feature_name)

        # validate index
        if start_idx is None:
            if self.train_range == None or self.test_range == None:
                self.start_idx = np.random.randint(self.window_size - 1, len(self.df) - self._cfg.eps_length)
            elif self._env_id[-1] == 'e':
                boundary = int(len(self.df) * (1 + self.test_range))
                assert len(self.df) - self._cfg.eps_length > boundary + self.window_size,\
                 "parameter test_range is too large!"
                self.start_idx = np.random.randint(boundary + self.window_size, len(self.df) - self._cfg.eps_length)
            else:
                boundary = int(len(self.df) * self.train_range)
                assert boundary - self._cfg.eps_length > self.window_size,\
                 "parameter test_range is too small!"
                self.start_idx = np.random.randint(self.window_size, boundary - self._cfg.eps_length)
        else:
            self.start_idx = start_idx

        self._start_tick = self.start_idx
        self._end_tick = self._start_tick + self._cfg.eps_length - 1

        return prices, selected_feature, feature_dim_len

    # override
    def _calculate_reward(self, action: int) -> np.float32:
        step_reward = 0.
        current_price = (self.raw_prices[self._current_tick])
        last_trade_price = (self.raw_prices[self._last_trade_tick])
        ratio = current_price / last_trade_price
        cost = np.log((1 - self.trade_fee_ask_percent) * (1 - self.trade_fee_bid_percent))

        if action == Actions.BUY and self._position == Positions.SHORT:
            step_reward = np.log(2 - ratio) + cost

        if action == Actions.SELL and self._position == Positions.LONG:
            step_reward = np.log(ratio) + cost

        if action == Actions.DOUBLE_SELL and self._position == Positions.LONG:
            step_reward = np.log(ratio) + cost

        if action == Actions.DOUBLE_BUY and self._position == Positions.SHORT:
            step_reward = np.log(2 - ratio) + cost

        step_reward = float(step_reward)

        return step_reward

    # override
    def max_possible_profit(self) -> float:
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.

        while current_tick <= self._end_tick:

            if self.raw_prices[current_tick] < self.raw_prices[current_tick - 1]:
                while (current_tick <= self._end_tick
                       and self.raw_prices[current_tick] < self.raw_prices[current_tick - 1]):
                    current_tick += 1

                current_price = self.raw_prices[current_tick - 1]
                last_trade_price = self.raw_prices[last_trade_tick]
                tmp_profit = profit * (2 - (current_price / last_trade_price)) * (1 - self.trade_fee_ask_percent
                                                                                  ) * (1 - self.trade_fee_bid_percent)
                profit = max(profit, tmp_profit)
            else:
                while (current_tick <= self._end_tick
                       and self.raw_prices[current_tick] >= self.raw_prices[current_tick - 1]):
                    current_tick += 1

                current_price = self.raw_prices[current_tick - 1]
                last_trade_price = self.raw_prices[last_trade_tick]
                tmp_profit = profit * (current_price / last_trade_price) * (1 - self.trade_fee_ask_percent
                                                                            ) * (1 - self.trade_fee_bid_percent)
                profit = max(profit, tmp_profit)
            last_trade_tick = current_tick - 1

        return profit

    def __repr__(self) -> str:
        return "DI-engine Stocks Trading Env"
