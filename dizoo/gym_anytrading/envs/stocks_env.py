from typing import Any
import numpy as np
from dizoo.gym_anytrading.envs.trading_env import TradingEnv, Actions, Positions
from ding.utils import ENV_REGISTRY
from ding.torch_utils import to_ndarray


@ENV_REGISTRY.register('stocks-v0')
class StocksEnv(TradingEnv):

    def __init__(self,cfg):

        super().__init__(cfg)

        self.trade_fee_bid_percent = 0.01  # unit
        self.trade_fee_ask_percent = 0.005  # unit


    def _process_data(self, start_idx :int = None) -> Any:
        '''
        Overview:
            used by env.reset(), process the raw data.
        Arguments:
            - start_idx (int): the start tick; if None, then randomly select.
        Returns:
            - prices: the close.
            - signal_features: feature map
        '''
        prices = self.df.loc[:, 'Close'].to_numpy()
        diff = np.insert(np.diff(prices), 0, 0)
        opens = self.df.loc[:, 'Open'].to_numpy()
        highs = self.df.loc[:, 'High'].to_numpy()
        lows = self.df.loc[:, 'Low'].to_numpy()
        adjclose = self.df.loc[:, 'Adj Close'].to_numpy()
        volumes = self.df.loc[:, 'Volume'].to_numpy()


        # validate index 
        if start_idx == None:
            self.start_idx = np.random.randint(self.window_size, len(self.df) - self._cfg.eps_length-1)
        else:
            self.start_idx = start_idx
        
        self._start_tick = self.start_idx
        self._end_tick = self._start_tick + self._cfg.eps_length - 1
        
        
        signal_features = np.column_stack((prices, diff, volumes))
        # signal_features = np.column_stack((prices, opens, highs, lows, adjclose, volumes))
        # signal_features = np.column_stack((prices, diff, opens, highs, lows, adjclose, volumes))
        
        return prices, signal_features


    def _calculate_reward(self, action: int) -> np.float32:
        step_reward = 0.
        current_price = (self.raw_prices[self._current_tick])
        last_trade_price = (self.raw_prices[self._last_trade_tick])
        ratio = current_price/last_trade_price
        cost = np.log((1 - self.trade_fee_ask_percent)*(1 - self.trade_fee_bid_percent))

        if action == Actions.BUY.value and self._position == Positions.SHORT:
            step_reward = np.log(2-ratio) + cost
        
        if action == Actions.SELL.value and self._position == Positions.LONG:
            step_reward = np.log(ratio) + cost

        if action == Actions.DOUBLE_SELL.value and self._position == Positions.LONG:
            step_reward = np.log(ratio) + cost

        if action == Actions.DOUBLE_BUY.value and self._position == Positions.SHORT:
            step_reward = np.log(2-ratio) + cost

        
        step_reward = to_ndarray([step_reward]).astype(np.float32)
        
        return step_reward



    def max_possible_profit(self) -> float:
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
                tmp_profit = profit *(2 - (current_price / last_trade_price) )* (1 - self.trade_fee_ask_percent) * (1 - self.trade_fee_bid_percent)
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

        return profit

    def __repr__(self) -> str:
        return "DI-engine Stocks Trading Env"


