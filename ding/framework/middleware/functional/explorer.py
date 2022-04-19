from typing import TYPE_CHECKING, Callable, List
from easydict import EasyDict
from ding.rl_utils import get_epsilon_greedy_fn

if TYPE_CHECKING:
    from ding.framework import Context


def eps_greedy_handler(cfg: EasyDict) -> Callable:
    eps_cfg = cfg.policy.other.eps
    handle = get_epsilon_greedy_fn(eps_cfg.start, eps_cfg.end, eps_cfg.decay, eps_cfg.type)

    def _eps_greedy(ctx: "Context"):
        ctx.collect_kwargs['eps'] = handle(ctx.env_step)
        yield
        try:
            ctx.collect_kwargs.pop('eps')
        except:  # noqa
            pass

    return _eps_greedy
