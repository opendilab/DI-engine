from typing import TYPE_CHECKING, Callable, List
from easydict import EasyDict
from ding.rl_utils import get_epsilon_greedy_fn

if TYPE_CHECKING:
    from ding.framework import OnlineRLContext


def eps_greedy_handler(cfg: EasyDict) -> Callable:
    """
    Overview:
        The middleware that computes epsilon value according to the env_step.
    Arguments:
        - cfg (:obj:`EasyDict`): Config.
    """

    eps_cfg = cfg.policy.other.eps
    handle = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)

    def _eps_greedy(ctx: "OnlineRLContext"):
        """
        Input of ctx:
            - env_step (:obj:`int`): The env steps count.
        Output of ctx:
            - collect_kwargs['eps'] (:obj:`float`): The eps conditioned on env_step and cfg.
        """

        ctx.collect_kwargs['eps'] = handle(ctx.env_step)
        yield
        try:
            ctx.collect_kwargs.pop('eps')
        except:  # noqa
            pass

    return _eps_greedy


def eps_greedy_masker():
    """
    Overview:
        The middleware that returns masked epsilon value and stop generating \
             actions by the e_greedy method.
    """

    def _masker(ctx: "OnlineRLContext"):
        """
        Output of ctx:
            - collect_kwargs['eps'] (:obj:`float`): The masked eps value, default to -1.
        """

        ctx.collect_kwargs['eps'] = -1

    return _masker
