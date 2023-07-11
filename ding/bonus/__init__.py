import ding.config
from .a2c import A2CAgent
from .c51 import C51Agent
from .ddpg import DDPGAgent
from .dqn import DQNAgent
from .pg import PGAgent
from .ppof import PPOF
from .ppo_offpolicy import PPOOffPolicyAgent
from .sac import SACAgent
from .sql import SQLAgent
from .td3 import TD3Agent

supported_algo = dict(
    A2C=A2CAgent,
    C51=C51Agent,
    DDPG=DDPGAgent,
    DQN=DQNAgent,
    PG=PGAgent,
    PPOF=PPOF,
    PPOOffPolicy=PPOOffPolicyAgent,
    SAC=SACAgent,
    SQL=SQLAgent,
    TD3=TD3Agent,
)

supported_algo_list = list(supported_algo.keys())


def env_supported(algo: str = None) -> list:
    """
    return list of the envs that supported by di-engine.
    """

    if algo is not None:
        if algo.upper() == "A2C":
            return list(ding.config.A2C.supported_env.keys())
        elif algo.upper() == "C51":
            return list(ding.config.C51.supported_env.keys())
        elif algo.upper() == "DDPG":
            return list(ding.config.DDPG.supported_env.keys())
        elif algo.upper() == "DQN":
            return list(ding.config.DQN.supported_env.keys())
        elif algo.upper() == "PG":
            return list(ding.config.PG.supported_env.keys())
        elif algo.upper() == "PPOF":
            return list(ding.config.PPOF.supported_env.keys())
        elif algo.upper() == "PPOOFFPOLICY":
            return list(ding.config.PPOOffPolicy.supported_env.keys())
        elif algo.upper() == "SAC":
            return list(ding.config.SAC.supported_env.keys())
        elif algo.upper() == "SQL":
            return list(ding.config.SQL.supported_env.keys())
        elif algo.upper() == "TD3":
            return list(ding.config.TD3.supported_env.keys())
        else:
            raise ValueError("The algo {} is not supported by di-engine.".format(algo))
    else:
        supported_env = set()
        supported_env.update(ding.config.A2C.supported_env.keys())
        supported_env.update(ding.config.C51.supported_env.keys())
        supported_env.update(ding.config.DDPG.supported_env.keys())
        supported_env.update(ding.config.DQN.supported_env.keys())
        supported_env.update(ding.config.PG.supported_env.keys())
        supported_env.update(ding.config.PPOF.supported_env.keys())
        supported_env.update(ding.config.PPOOffPolicy.supported_env.keys())
        supported_env.update(ding.config.SAC.supported_env.keys())
        supported_env.update(ding.config.SQL.supported_env.keys())
        supported_env.update(ding.config.TD3.supported_env.keys())
        # return the list of the envs
        return list(supported_env)


supported_env = env_supported()


def algo_supported(env_id: str = None) -> list:
    """
    return list of the algos that supported by di-engine.
    """
    if env_id is not None:
        algo = []
        if env_id.upper() in [item.upper() for item in ding.config.A2C.supported_env.keys()]:
            algo.append("A2C")
        if env_id.upper() in [item.upper() for item in ding.config.C51.supported_env.keys()]:
            algo.append("C51")
        if env_id.upper() in [item.upper() for item in ding.config.DDPG.supported_env.keys()]:
            algo.append("DDPG")
        if env_id.upper() in [item.upper() for item in ding.config.DQN.supported_env.keys()]:
            algo.append("DQN")
        if env_id.upper() in [item.upper() for item in ding.config.PG.supported_env.keys()]:
            algo.append("PG")
        if env_id.upper() in [item.upper() for item in ding.config.PPOF.supported_env.keys()]:
            algo.append("PPOF")
        if env_id.upper() in [item.upper() for item in ding.config.PPOOffPolicy.supported_env.keys()]:
            algo.append("PPOOffPolicy")
        if env_id.upper() in [item.upper() for item in ding.config.SAC.supported_env.keys()]:
            algo.append("SAC")
        if env_id.upper() in [item.upper() for item in ding.config.SQL.supported_env.keys()]:
            algo.append("SQL")
        if env_id.upper() in [item.upper() for item in ding.config.TD3.supported_env.keys()]:
            algo.append("TD3")

        if len(algo) == 0:
            raise ValueError("The env {} is not supported by di-engine.".format(env_id))
        return algo
    else:
        return supported_algo_list


def is_supported(env_id: str = None, algo: str = None) -> bool:
    """
    Check if the env-algo pair is supported by di-engine.
    """
    if env_id is not None and env_id.upper() in [item.upper() for item in supported_env.keys()]:
        if algo is not None and algo.upper() in supported_algo_list:
            if env_id.upper() in env_supported(algo):
                return True
            else:
                return False
        elif algo is None:
            return True
        else:
            return False
    elif env_id is None:
        if algo is not None and algo.upper() in supported_algo_list:
            return True
        elif algo is None:
            raise ValueError("Please specify the env or algo.")
        else:
            return False
    else:
        return False
