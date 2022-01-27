from typing import TYPE_CHECKING, Callable
import time
import math
if TYPE_CHECKING:
    from ding.framework import Task, Context


def pace_controller(
        task: "Task",
        theme: str = "",
        identity: str = "",
        timeout: float = math.inf,
) -> Callable:
    """
    Overview:
        The pace controller middleware provides ways to block current thread routine until \
            event of same theme being called from different identity or time being out.
    Arguments:
        - task (:obj:`Task`): Task in which the pace controller being applied.
        - theme (:obj:`str`): Theme string is a common string, \
            which is subscribed by all pace controller that work together.
        - identity (:obj:`str`): Identity string determines the identity of a pace controller.
            Event from pace controller of same identity will be neglected.
            Pace controller with empty string identity are recognized as a unique one.
        - timeout (:obj:`float`): Timeout in seconds.
    Returns:
        - _pace_control (:obj:`Callable`): The wrapper function for pace controller.
    """
    time.sleep(1)
    event_theme = "_pace_control_{}".format(theme)
    event_received = False
    _identity = identity

    def _event_received(another_identity) -> None:
        nonlocal event_received, _identity
        if another_identity != _identity or _identity == "":
            event_received = True
        return

    if task.router.is_active:
        task.on(event_theme, _event_received)

    def _pace_control(ctx: "Context") -> None:
        """
        Overview:
            Wait for an event and block current thread until event of same theme being called \
                from different identity or time being out.
        Arguments:
            - ctx (:obj:`Context`): Context of task object that using the current pace controller.
        """
        nonlocal event_received
        if task.router.is_active:
            task.emit(event_theme, identity, only_remote=True)
            time_begin = time.time()
            while not event_received and not task.finish:
                time.sleep(0.01)
                if (time.time() - time_begin) > timeout:
                    break
            event_received = False
        return

    return _pace_control
