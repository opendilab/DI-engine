import pytest

from .bases import _TestInteractionBase, _random_slave_channel_and_port, _slave_endpoint, _get_master_endpoint
from ..test_utils import random_port, random_channel
from ...exception import SlaveErrorCode, SlaveChannelInvalid


@pytest.mark.unittest
class TestInteractionErrors(_TestInteractionBase):

    @pytest.mark.execution_timeout(20.0, method='thread')
    def test_slave_simple_connection(self):
        _slave_port, _slave_channel = _random_slave_channel_and_port()
        slave_thread, open_slave_event, close_slave_event = _slave_endpoint(_slave_port, _slave_channel)

        slave_thread.start()
        open_slave_event.wait()

        try:
            _master_port = random_port()
            _master_channel = random_channel(excludes=[_slave_channel])
            master = _get_master_endpoint(_master_port, _master_channel)
            with master:
                assert master.ping()

                with pytest.raises(SlaveChannelInvalid) as ei:
                    with master.new_connection('conn', '127.0.0.1', _slave_port):
                        pytest.fail('Should not reach here!')

                err = ei.value
                assert not err.success
                assert err.status_code == 403
                assert err.code == SlaveErrorCode.CHANNEL_INVALID

                assert 'conn' not in master
        finally:
            close_slave_event.set()
            slave_thread.join()
