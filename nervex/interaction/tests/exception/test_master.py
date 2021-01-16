import pytest

from ...exception.master import MasterErrorCode, get_exception_class_by_error_code, MasterSuccess, \
    MasterSystemShuttingDown, MasterTaskDataInvalid, MasterSlaveTokenNotGiven, MasterSlaveTokenInvalid, \
    MasterSelfTokenNotGiven, MasterSelfTokenInvalid, MasterChannelInvalid, \
    MasterChannelNotGiven, MasterMasterTokenInvalid, MasterMasterTokenNotGiven


@pytest.mark.unittest
class TestInteractionExceptionMaster:

    def test_error_code(self):
        assert len(MasterErrorCode.__members__) == 11
        assert MasterErrorCode.SUCCESS == 0

    def test_exception_class(self):
        assert get_exception_class_by_error_code(MasterErrorCode.SUCCESS) == MasterSuccess

        assert get_exception_class_by_error_code(MasterErrorCode.SYSTEM_SHUTTING_DOWN) == MasterSystemShuttingDown

        assert get_exception_class_by_error_code(MasterErrorCode.CHANNEL_NOT_GIVEN) == MasterChannelNotGiven
        assert get_exception_class_by_error_code(MasterErrorCode.CHANNEL_INVALID) == MasterChannelInvalid

        assert get_exception_class_by_error_code(MasterErrorCode.MASTER_TOKEN_NOT_GIVEN) == MasterMasterTokenNotGiven
        assert get_exception_class_by_error_code(MasterErrorCode.MASTER_TOKEN_INVALID) == MasterMasterTokenInvalid

        assert get_exception_class_by_error_code(MasterErrorCode.SELF_TOKEN_NOT_GIVEN) == MasterSelfTokenNotGiven
        assert get_exception_class_by_error_code(MasterErrorCode.SELF_TOKEN_INVALID) == MasterSelfTokenInvalid

        assert get_exception_class_by_error_code(MasterErrorCode.SLAVE_TOKEN_NOT_GIVEN) == MasterSlaveTokenNotGiven
        assert get_exception_class_by_error_code(MasterErrorCode.SLAVE_TOKEN_INVALID) == MasterSlaveTokenInvalid

        assert get_exception_class_by_error_code(MasterErrorCode.TASK_DATA_INVALID) == MasterTaskDataInvalid
