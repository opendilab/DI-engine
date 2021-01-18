import pytest

from .test_base import _HTTPErrorGenerator
from ...exception.master import MasterErrorCode, \
    get_master_exception_class_by_error_code, get_master_exception_by_error, MasterSuccess, \
    MasterSystemShuttingDown, MasterTaskDataInvalid, MasterSlaveTokenNotGiven, MasterSlaveTokenInvalid, \
    MasterSelfTokenNotGiven, MasterSelfTokenInvalid, MasterChannelInvalid, \
    MasterChannelNotGiven, MasterMasterTokenInvalid, MasterMasterTokenNotGiven


@pytest.mark.unittest
class TestInteractionExceptionMaster(_HTTPErrorGenerator):

    def test_error_code(self):
        assert len(MasterErrorCode.__members__) == 11
        assert MasterErrorCode.SUCCESS == 0

    def test_exception_class(self):
        assert get_master_exception_class_by_error_code(MasterErrorCode.SUCCESS) == MasterSuccess

        assert get_master_exception_class_by_error_code(
            MasterErrorCode.SYSTEM_SHUTTING_DOWN
        ) == MasterSystemShuttingDown

        assert get_master_exception_class_by_error_code(MasterErrorCode.CHANNEL_NOT_GIVEN) == MasterChannelNotGiven
        assert get_master_exception_class_by_error_code(MasterErrorCode.CHANNEL_INVALID) == MasterChannelInvalid

        assert get_master_exception_class_by_error_code(
            MasterErrorCode.MASTER_TOKEN_NOT_GIVEN
        ) == MasterMasterTokenNotGiven
        assert get_master_exception_class_by_error_code(
            MasterErrorCode.MASTER_TOKEN_INVALID
        ) == MasterMasterTokenInvalid

        assert get_master_exception_class_by_error_code(MasterErrorCode.SELF_TOKEN_NOT_GIVEN) == MasterSelfTokenNotGiven
        assert get_master_exception_class_by_error_code(MasterErrorCode.SELF_TOKEN_INVALID) == MasterSelfTokenInvalid

        assert get_master_exception_class_by_error_code(
            MasterErrorCode.SLAVE_TOKEN_NOT_GIVEN
        ) == MasterSlaveTokenNotGiven
        assert get_master_exception_class_by_error_code(MasterErrorCode.SLAVE_TOKEN_INVALID) == MasterSlaveTokenInvalid

        assert get_master_exception_class_by_error_code(MasterErrorCode.TASK_DATA_INVALID) == MasterTaskDataInvalid

    def test_get_master_exception_by_error(self):
        err = get_master_exception_by_error(self._generate_exception(101, 'This is system shutting down.'))
        assert isinstance(err, MasterSystemShuttingDown)
        assert not err.success
        assert err.status_code == 400
        assert err.code == 101
        assert err.message == 'This is system shutting down.'
        assert err.data == {}

        err = get_master_exception_by_error(self._generate_exception(601, 'Task data invalid.', data={'value': 233}))
        assert isinstance(err, MasterTaskDataInvalid)
        assert not err.success
        assert err.status_code == 400
        assert err.code == 601
        assert err.message == 'Task data invalid.'
        assert err.data == {'value': 233}
