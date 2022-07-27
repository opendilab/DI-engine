from easydict import EasyDict
from dizoo.gym_anytrading.envs.stocks_env import StocksEnv

if __name__ == "__main__":
    cfg = EasyDict({"env_id": 'stocks-v0', "eps_length": 2334, "window_size": 20})
    env = StocksEnv(cfg)
    env.reset(1)
    print(env.max_possible_profit())
