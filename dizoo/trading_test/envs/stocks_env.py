import sys
sys.path.append( '/home/PJLAB/chenyun/trade_test/DI-engine')
from cmath import cos
import numpy as np

from .trading_env import TradingEnv, Actions, Positions
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY
from ding.torch_utils import to_ndarray


@ENV_REGISTRY.register('stocks-v0')
class StocksEnv(TradingEnv):

    def __init__(self,cfg):

        super().__init__(cfg)

        self.trade_fee_bid_percent = 0.01  # unit
        self.trade_fee_ask_percent = 0.005  # unit


    def _process_data(self, start_idx = None):
        opens = self.df.loc[:, 'Open'].to_numpy()
        highs = self.df.loc[:, 'High'].to_numpy()
        lows = self.df.loc[:, 'Low'].to_numpy()
        prices = self.df.loc[:, 'Close'].to_numpy()
        adjclose = self.df.loc[:, 'Adj Close'].to_numpy()
        volumes = self.df.loc[:, 'Volume'].to_numpy()

        #prices[self.frame_bound[0] - self.window_size]  # validate index (TODO: Improve validation)
        if start_idx == None:
            self.start_idx = np.random.randint(self.window_size, len(self.df) - self._cfg.eps_length-1)
        else:
            self.start_idx = start_idx
        #prices = prices[self.start_idx-self.window_size: self.start_idx + self._cfg.eps_length+1]
        self._start_tick = self.start_idx
        self._end_tick = self._start_tick + self._cfg.eps_length - 1
        diff = np.insert(np.diff(prices), 0, 0)
        # signal_features = np.column_stack((prices, diff, opens, highs, lows, adjclose, volumes))
        signal_features = np.column_stack((prices, diff, volumes))
        # signal_features = np.column_stack((prices, opens, highs, lows, adjclose, volumes))
        
        return prices, signal_features


    def _calculate_reward(self, action):
        step_reward = 0.
        current_price = np.log(self.raw_prices[self._current_tick])
        last_trade_price = np.log(self.raw_prices[self._last_trade_tick])
        cost = np.log((1 - self.trade_fee_ask_percent)*(1 - self.trade_fee_bid_percent))
        if (action == Actions.Buy.value and self._position == Positions.Short):
            step_reward += last_trade_price - current_price + cost



        
        if (action == Actions.Sell.value and self._position == Positions.Long):
            step_reward += current_price - last_trade_price + cost

        
        
        step_reward = to_ndarray([step_reward]).astype(np.float32)
        return step_reward


    def _update_profit(self, action):
        current_price = self.raw_prices[self._current_tick]
        last_trade_price = self.raw_prices[self._last_trade_tick]
        if (action == Actions.Buy.value and self._position == Positions.Short):
            self._total_profit = self._total_profit*(1 - self.trade_fee_ask_percent)*\
                (1 - self.trade_fee_bid_percent)*(last_trade_price/current_price)



        if (action == Actions.Sell.value and self._position == Positions.Long):
            self._total_profit = self._total_profit*(1 - self.trade_fee_ask_percent)*\
                (1 - self.trade_fee_bid_percent)*(current_price/last_trade_price)




    def max_possible_profit(self):
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.

        while current_tick <= self._end_tick:
            
            if self.raw_prices[current_tick] < self.raw_prices[current_tick - 1]:
                while (current_tick <= self._end_tick and
                       self.raw_prices[current_tick] < self.raw_prices[current_tick - 1]):
                    current_tick += 1
                
                current_price = self.raw_prices[current_tick - 1]
                last_trade_price = self.raw_prices[last_trade_tick]
                tmp_profit = profit *(last_trade_price / current_price )* (1 - self.trade_fee_ask_percent) * (1 - self.trade_fee_bid_percent)
                profit = max(profit, tmp_profit)
            else:
                while (current_tick <= self._end_tick and
                       self.raw_prices[current_tick] >= self.raw_prices[current_tick - 1]):
                    current_tick += 1
                
                current_price = self.raw_prices[current_tick - 1]
                last_trade_price = self.raw_prices[last_trade_tick]
                tmp_profit = profit * (current_price/ last_trade_price) * (1 - self.trade_fee_ask_percent) * (1 - self.trade_fee_bid_percent)
                profit = max(profit, tmp_profit)
            last_trade_tick = current_tick - 1

        return np.log(profit)

    def __repr__(self) -> str:
        return "DI-engine Stocks Trading Env"


