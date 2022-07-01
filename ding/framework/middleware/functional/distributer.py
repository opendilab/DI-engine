from time import sleep, time
from typing import TYPE_CHECKING
from ditk import logging
from ding.framework import task
if TYPE_CHECKING:
    from ding.framework.context import Context
    from torch.nn import Module


def model_exchanger(model: "Module"):
    """
    Overview:
        Exchange model between processes, only the learner will send the model,
        otherwise the model will only be received.
        If you are using a shared model on a single host, there is no need to use this middleware.
    Arguments:
        - model (:obj:`torch.nn.Module`): Pytorch module.
    """
    if not task.router.is_active:
        return task.void()

    if len(task.roles) == 0:
        logging.warning("The task does not have any roles defined, the model_exchanger will not work.")
        return task.void()

    event_name = "model_exchanger"
    state_dict_cache = None

    is_learner = task.has_role(task.role.LEARNER)
    if not is_learner:

        def cache_state_dict(state_dict):
            nonlocal state_dict_cache
            state_dict_cache = state_dict

        task.on(event_name, cache_state_dict)

    def _model_exchanger(ctx: "Context"):
        if not is_learner:
            if ctx.total_step != 0:  # Skip first iteration
                nonlocal state_dict_cache
                start = time()
                while True:
                    if time() - start > 60:
                        logging.warning("Timeout when waiting for new model! Node id: {}".format(task.router.node_id))
                        break
                    if state_dict_cache is None:
                        sleep(0.01)
                    else:
                        model.load_state_dict(state_dict_cache)
                        state_dict_cache = None
                        break

        if is_learner:
            yield
            task.emit(event_name, model.state_dict(), only_remote=True)

    return _model_exchanger
