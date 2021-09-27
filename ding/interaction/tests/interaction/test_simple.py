import pytest
from requests import HTTPError

from .bases import _TestInteractionBase, _random_slave_channel_and_port, _slave_endpoint, _get_master_endpoint
from ..test_utils import random_port
from ...master.task import TaskStatus


@pytest.mark.unittest
class TestInteractionSimple(_TestInteractionBase):

    @pytest.mark.execution_timeout(10.0, method='thread')
    def test_slave_launch(self):
        _slave_port, _channel = _random_slave_channel_and_port()
        slave_thread, open_slave_event, close_slave_event = _slave_endpoint(_slave_port, _channel)

        slave_thread.start()
        open_slave_event.wait()

        close_slave_event.set()
        slave_thread.join()

    @pytest.mark.execution_timeout(20.0, method='thread')
    def test_slave_simple_connection(self):
        _slave_port, _channel = _random_slave_channel_and_port()
        slave_thread, open_slave_event, close_slave_event = _slave_endpoint(_slave_port, _channel)

        slave_thread.start()
        open_slave_event.wait()

        try:
            _master_port = random_port()
            master = _get_master_endpoint(_master_port, _channel)
            with master:
                assert master.ping()

                with master.new_connection('conn', '127.0.0.1', _slave_port) as conn:
                    assert conn.is_connected
                    assert 'conn' in master
                    assert master['conn'] == conn

                assert not conn.is_connected
                assert 'conn' not in master

                conn = master.new_connection('conn', '127.0.0.1', _slave_port)
                conn.connect()
                assert conn.is_connected
                assert 'conn' in master
                assert master['conn'] == conn
                conn.disconnect()
                assert not conn.is_connected
                assert 'conn' not in master

                conn = master.new_connection('conn', '127.0.0.1', _slave_port)
                conn.connect()
                assert conn.is_connected
                assert 'conn' in master
                assert master['conn'] == conn
                del master['conn']
                assert not conn.is_connected
                assert 'conn' not in master

        finally:
            close_slave_event.set()
            slave_thread.join()

    @pytest.mark.execution_timeout(20.0, method='thread')
    def test_slave_simple_task(self):
        _slave_port, _channel = _random_slave_channel_and_port()
        slave_thread, open_slave_event, close_slave_event = _slave_endpoint(_slave_port, _channel)

        slave_thread.start()
        open_slave_event.wait()

        try:
            _master_port = random_port()
            master = _get_master_endpoint(_master_port, _channel)
            with master:
                with master.new_connection('conn', '127.0.0.1', _slave_port) as conn:
                    task = conn.new_task({'a': 2, 'b': 3})
                    task.start().join()

                    assert task.result == {'sum': 5}
                    assert task.status == TaskStatus.COMPLETED

                    _res_1, _res_2, _res_3 = None, None, None

                    def _set_res_1(t, r):
                        nonlocal _res_1
                        _res_1 = r['sum']

                    def _set_res_2(t, r):
                        nonlocal _res_2
                        _res_2 = r

                    def _set_res_3(t, r):
                        nonlocal _res_3
                        _res_3 = r

                    task = conn.new_task({'a': 2, 'b': 3}) \
                        .on_complete(_set_res_1).on_complete(_set_res_2) \
                        .on_fail(_set_res_3)
                    task.start().join()

                    assert task.result == {'sum': 5}
                    assert task.status == TaskStatus.COMPLETED
                    assert _res_1 == 5
                    assert _res_2 == {'sum': 5}
                    assert _res_3 is None

                    _res_1, _res_2, _res_3 = None, None, None
                    task = conn.new_task({'a': 2, 'bb': 3}) \
                        .on_complete(_set_res_1).on_complete(_set_res_2) \
                        .on_fail(_set_res_3)
                    task.start().join()

                    assert task.result == {'message': 'ab not found'}
                    assert task.status == TaskStatus.FAILED
                    assert _res_1 is None
                    assert _res_2 is None
                    assert _res_3 == {'message': 'ab not found'}
        except HTTPError as err:
            print(err.response)
            print(err.response.content)
            print(err.request)

            raise err
        finally:
            close_slave_event.set()
            slave_thread.join()
