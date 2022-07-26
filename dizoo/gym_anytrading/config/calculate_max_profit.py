from easydict import EasyDict
from dizoo.trading_test.envs.stocks_env import StocksEnv


if __name__ == "__main__":
    cfg = EasyDict({"env_id": [0], "eps_length": 2334, "window_size": 20})
    env = StocksEnv(cfg)
    env.reset(1)
    print(env.max_possible_profit())
