from typing import TYPE_CHECKING
import time
import math
if TYPE_CHECKING:
    from ding.framework import Task, Context


def Pace_Controller(
    task: "Task",
    theme: str = "unknown",
    active_state: bool = True,
    filter_the_similar: bool = True,
    start_up_delay: float = 0.5,
    timeout: float = math.inf,
    emit_option: bool = True
):

    if start_up_delay > 0:
        time.sleep(start_up_delay)

    random_event = "pace_control_{}".format(theme)
    event_received = False
    all_middleware_names = list(map(lambda fn: getattr(fn, "__name__", type(fn).__name__), task.middleware))

    def _event_received(node_id, middleware_names):
        nonlocal event_received
        not_same = map(lambda name: name not in all_middleware_names, middleware_names)

        if filter_the_similar:
            not_same = any(not_same)
        else:
            not_same = True

        if task.router.is_active:
            if task.router.node_id != node_id and not_same:
                event_received = True
        else:
            if not_same:
                event_received = True

    if active_state:
        task.on(random_event, _event_received)

    def pace_locking():
        nonlocal event_received
        time_begin = time.time()
        while not event_received and not task.finish:
            time.sleep(0.01)
            if (time.time() - time_begin) > timeout:
                break
        event_received = False

    def _pace_control(ctx: "Context"):
        if emit_option:
            if task.router.is_active:
                task.emit(random_event, task.router.node_id, all_middleware_names, only_remote=True)
            else:
                task.emit(random_event, task.router.node_id, all_middleware_names, only_local=True)
        if active_state:
            pace_locking()

    return _pace_control