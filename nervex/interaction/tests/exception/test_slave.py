import pytest

from .test_base import _HTTPErrorGenerator
from ...exception.slave import SlaveErrorCode, \
    get_slave_exception_class_by_error_code, get_slave_exception_by_error, SlaveSystemShuttingDown, \
    SlaveSlaveConnectionRefused, SlaveSlaveDisconnectionRefused, SlaveSlaveNotConnected, SlaveSlaveAlreadyConnected, \
    SlaveTaskRefused, SlaveMasterTokenInvalid, SlaveMasterTokenNotFound, SlaveSelfTokenNotFound, \
    SlaveTaskAlreadyExist, SlaveSelfTokenInvalid, SlaveChannelNotFound, SlaveChannelInvalid, SlaveSuccess


@pytest.mark.unittest
class TestInteractionExceptionSlave(_HTTPErrorGenerator):

    def test_error_code(self):
        assert len(SlaveErrorCode.__members__) == 14
        assert SlaveErrorCode.SUCCESS == 0

    # noinspection DuplicatedCode
    def test_exception_class(self):
        assert get_slave_exception_class_by_error_code(SlaveErrorCode.SUCCESS) == SlaveSuccess

        assert get_slave_exception_class_by_error_code(SlaveErrorCode.SYSTEM_SHUTTING_DOWN) == SlaveSystemShuttingDown

        assert get_slave_exception_class_by_error_code(SlaveErrorCode.CHANNEL_NOT_FOUND) == SlaveChannelNotFound
        assert get_slave_exception_class_by_error_code(SlaveErrorCode.CHANNEL_INVALID) == SlaveChannelInvalid

        assert get_slave_exception_class_by_error_code(
            SlaveErrorCode.MASTER_TOKEN_NOT_FOUND
        ) == SlaveMasterTokenNotFound
        assert get_slave_exception_class_by_error_code(SlaveErrorCode.MASTER_TOKEN_INVALID) == SlaveMasterTokenInvalid

        assert get_slave_exception_class_by_error_code(SlaveErrorCode.SELF_TOKEN_NOT_FOUND) == SlaveSelfTokenNotFound
        assert get_slave_exception_class_by_error_code(SlaveErrorCode.SELF_TOKEN_INVALID) == SlaveSelfTokenInvalid

        assert get_slave_exception_class_by_error_code(
            SlaveErrorCode.SLAVE_ALREADY_CONNECTED
        ) == SlaveSlaveAlreadyConnected
        assert get_slave_exception_class_by_error_code(SlaveErrorCode.SLAVE_NOT_CONNECTED) == SlaveSlaveNotConnected
        assert get_slave_exception_class_by_error_code(
            SlaveErrorCode.SLAVE_CONNECTION_REFUSED
        ) == SlaveSlaveConnectionRefused
        assert get_slave_exception_class_by_error_code(
            SlaveErrorCode.SLAVE_DISCONNECTION_REFUSED
        ) == SlaveSlaveDisconnectionRefused

        assert get_slave_exception_class_by_error_code(SlaveErrorCode.TASK_ALREADY_EXIST) == SlaveTaskAlreadyExist
        assert get_slave_exception_class_by_error_code(SlaveErrorCode.TASK_REFUSED) == SlaveTaskRefused

    def test_get_slave_exception_by_error(self):
        err = get_slave_exception_by_error(self._generate_exception(101, 'This is slave shutting down.'))
        assert isinstance(err, SlaveSystemShuttingDown)
        assert not err.success
        assert err.status_code == 400
        assert err.code == 101
        assert err.message == 'This is slave shutting down.'
        assert err.data == {}

        err = get_slave_exception_by_error(self._generate_exception(602, 'Task refused.', data={'value': 233}))
        assert isinstance(err, SlaveTaskRefused)
        assert not err.success
        assert err.status_code == 400
        assert err.code == 602
        assert err.message == 'Task refused.'
        assert err.data == {'value': 233}
