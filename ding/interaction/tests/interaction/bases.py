from multiprocessing import Event, Process
from typing import Mapping, Any, Tuple

from ..test_utils import silence_function, random_channel, random_port
from ...master import Master
from ...slave import Slave, TaskFail


class _TestInteractionBase:
    # noinspection PyMethodMayBeStatic
    def _slave_endpoint(self, port: int, channel: int, silence: bool = True):
        open_slave_event = Event()
        close_slave_event = Event()

        class MySlave(Slave):

            def _process_task(self, task: Mapping[str, Any]):
                if 'a' in task.keys() and 'b' in task.keys():
                    return {'sum': task['a'] + task['b']}
                else:
                    raise TaskFail(result={'message': 'ab not found'}, message='A or B not found in task data.')

        def _run_slave():
            with MySlave('0.0.0.0', port, channel=channel):
                open_slave_event.set()
                close_slave_event.wait()

        if silence:
            _run_slave = silence_function()(_run_slave)

        slave_process = Process(target=_run_slave)

        return slave_process, open_slave_event, close_slave_event

    # noinspection PyMethodMayBeStatic
    def _get_master_endpoint(self, port: int, channel: int):

        class MyMaster(Master):
            pass

        return MyMaster('0.0.0.0', port, channel=channel)

    def _random_slave_channel_and_port(self) -> Tuple[int, int]:
        return random_port(), random_channel()
