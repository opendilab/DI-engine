import pytest
from requests import HTTPError

from .bases import _TestInteractionBase
from ..test_utils import random_port, random_channel
from ...base import get_values_from_response
from ...slave import SlaveErrorCode


@pytest.mark.unittest
class TestInteractionErrors(_TestInteractionBase):

    @pytest.mark.execution_timeout(20.0, method='thread')
    def test_slave_simple_connection(self):
        _slave_port, _slave_channel = self._random_slave_channel_and_port()
        slave_thread, open_slave_event, close_slave_event = self._slave_endpoint(_slave_port, _slave_channel)

        slave_thread.start()
        open_slave_event.wait()

        try:
            _master_port = random_port()
            _master_channel = random_channel(excludes=[_slave_channel])
            master = self._get_master_endpoint(_master_port, _master_channel)
            with master:
                assert master.ping()

                with pytest.raises(HTTPError) as ei:
                    with master.new_connection('conn', '127.0.0.1', _slave_port) as conn:
                        pytest.fail('Should not reach here!')

                err = ei.value
                _status_code, _success, _code, _, _ = get_values_from_response(err.response)
                assert _status_code == 403
                assert not _success
                assert _code == SlaveErrorCode.CHANNEL_INVALID

                assert 'conn' not in master
        finally:
            close_slave_event.set()
            slave_thread.join()
