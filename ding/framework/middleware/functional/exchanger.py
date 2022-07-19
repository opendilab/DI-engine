from ditk import logging
from queue import Empty
from typing import TYPE_CHECKING, List, Dict
from ding.framework import task
from ding.utils.data.structure.lifo_deque import LifoDeque
if TYPE_CHECKING:
    from ding.framework.context import Context
    from torch.nn import Module


def context_exchanger(send_keys: List[str] = None, recv_keys: List[str] = None, skip_n_iter: int = 0):
    """
    Overview:
        Send data from context in the backward stage.
        Buffer received data and wait if not get any data.
    Arguments:
        - send_keys (:obj:`List[str]`): Keys need to be sent.
        - recv_keys (:obj:`List[str]`): Keys need to be received.
        - skip_n_iter (:obj:`int`): Whether to skip the first N round of waiting,
            e.g. collecting data without waiting for a new model in the first N round,
            while training a model that needs to wait for data in the first round.
    """
    event_name = "context_exchanger"

    bufferd_payloads = LifoDeque(maxsize=100)
    task.on(event_name, lambda payload: bufferd_payloads.put(payload))

    def _context_exchanger(ctx: "Context"):
        if recv_keys:
            if ctx.total_step >= skip_n_iter:
                payload: Dict = bufferd_payloads.get()
                for key in recv_keys:
                    value = payload.get(key)
                    if value:
                        ctx[key] = value

        if send_keys:
            yield
            payload = {}
            for key in send_keys:
                payload[key] = ctx.get(key)
            if payload:
                task.emit(event_name, payload, only_remote=True)

    return _context_exchanger


def model_exchanger(model: "Module", is_learner: bool = False):
    """
    Overview:
        Exchange model between processes, only the learner will send the model,
        otherwise the model will only be received.
        If you are using a shared model on a single host, there is no need to use this middleware.
    Arguments:
        - model (:obj:`torch.nn.Module`): Pytorch module.
        - is_learner (:obj:`bool`): Whether use this middleware as learner or not.
    """
    event_name = "model_exchanger"
    bufferd_state_dict = LifoDeque(maxsize=1)

    if not is_learner:
        task.on(event_name, lambda state_dict: bufferd_state_dict.put(state_dict))

    def _model_exchanger(ctx: "Context"):
        if not is_learner:
            if ctx.total_step != 0:  # Skip first iteration
                try:
                    state_dict = bufferd_state_dict.get(timeout=5)
                    model.load_state_dict(state_dict)
                except Empty:
                    logging.warning("Timeout when waiting for new model!")

        if is_learner:
            yield
            task.emit(event_name, model.state_dict(), only_remote=True)

    return _model_exchanger
