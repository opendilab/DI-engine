from functools import partial
from multiprocessing import Event, Process
from typing import Mapping, Any, Tuple

from ..test_utils import silence_function, random_channel, random_port
from ...master import Master
from ...slave import Slave, TaskFail


class MySlave(Slave):

    def _process_task(self, task: Mapping[str, Any]):
        if 'a' in task.keys() and 'b' in task.keys():
            return {'sum': task['a'] + task['b']}
        else:
            raise TaskFail(result={'message': 'ab not found'}, message='A or B not found in task data.')


def _run_slave(port, channel, open_slave_event, close_slave_event):
    with MySlave('0.0.0.0', port, channel=channel):
        open_slave_event.set()
        close_slave_event.wait()


def _slave_endpoint(port: int, channel: int, silence: bool = False):
    open_slave_event = Event()
    close_slave_event = Event()

    _run = partial(_run_slave, port, channel, open_slave_event, close_slave_event)
    if silence:
        _run = silence_function()(_run)
    slave_process = Process(target=_run)

    return slave_process, open_slave_event, close_slave_event


class _MyMaster(Master):
    pass


def _get_master_endpoint(port: int, channel: int):
    return _MyMaster('0.0.0.0', port, channel=channel)


def _random_slave_channel_and_port() -> Tuple[int, int]:
    return random_port(), random_channel()


class _TestInteractionBase:
    pass
