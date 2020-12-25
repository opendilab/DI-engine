import pytest

from .bases import _TestInteractionBase
from ..test_utils import random_port


@pytest.mark.unittest
class TestInteractionErrors(_TestInteractionBase):
    @pytest.mark.execution_timeout(20.0, method='thread')
    def test_slave_simple_connection(self):
        _slave_port, _channel = self._random_slave_channel_and_port()
        slave_thread, open_slave_event, close_slave_event = self._slave_endpoint(_slave_port, _channel)

        slave_thread.start()
        open_slave_event.wait()

        try:
            _master_port = random_port()
            master = self._get_master_endpoint(_master_port, _channel)
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
