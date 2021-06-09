import warnings

try:
    from .gfootball_env import GfootballEnv
except ImportError:
    warnings.warn("not found gfootball env, please install it")
    GfootballEnv = None
