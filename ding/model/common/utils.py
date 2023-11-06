import copy
import torch
from easydict import EasyDict
from ding.utils import import_module, MODEL_REGISTRY


def create_model(cfg: EasyDict) -> torch.nn.Module:
    """
    Overview:
        Create a neural network model according to the given EasyDict-type ``cfg``.
    Arguments:
        - cfg: (:obj:`EasyDict`): User's model config. The key ``import_name`` is \
            used to import modules, and they key ``type`` is used to indicate the model.
    Returns:
        - (:obj:`torch.nn.Module`): The created neural network model.

    .. tip::
        This method will not modify the ``cfg`` , it will deepcopy the ``cfg`` and then modify it.
    """
    cfg = copy.deepcopy(cfg)
    import_module(cfg.pop('import_names', []))
    # here we must use the pop opeartion to ensure compatibility
    return MODEL_REGISTRY.build(cfg.pop("type"), **cfg)


def top_p_logits(logits, topp=0.9, filter_value=0, min_topk=1):
    """
    Filter a distribution of logits using nucleus (top-p) filtering
    https://github.com/OpenLMLab/MOSS/blob/e088f438d1a95d424c6dffef0d73134ebe62cb72/models_jittor/generation.py#L146
    """
    cum_logits = logits.clone()
    if topp > 0:
        logits_sorted, inds = torch.sort(logits, dim=-1, descending=True)
        mask = (logits_sorted.cumsum(dim=-1) - logits_sorted) >= topp
        mask[:, :min_topk] = False
        # Remove tokens with cumulative top_p above the threshold
        mask = torch.zeros_like(mask).to(torch.bool).scatter_(dim=-1, index=inds, src=mask)
        cum_logits[mask] = filter_value
        cum_logits.div_(cum_logits.sum(dim=-1, keepdim=True))
    return cum_logits
