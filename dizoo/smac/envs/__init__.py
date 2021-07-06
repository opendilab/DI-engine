import warnings

from .fake_smac_env import FakeSMACEnv
try:
    from .smac_env import SMACEnv
except ImportError:
    warnings.warn("not found pysc2 env, please install it")
    SMACEnv = None
