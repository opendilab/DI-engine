from typing import TYPE_CHECKING
from easydict import EasyDict
from ding.utils.data import default_collate

if TYPE_CHECKING:
    from ding.framework import OnlineRLContext


def obs_extractor(cfg: EasyDict, obs_model) -> None:

    def _extract(ctx: "OnlineRLContext"):
        if isinstance(ctx.train_data, list):  # not pre-processed data
            ctx.train_data = default_collate(ctx.train_data)
        ctx.train_data['obs'] = obs_model.encode(ctx.train_data['obs'])

    return _extract
