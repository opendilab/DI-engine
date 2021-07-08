import pytest

from ding.utils.system_helper import get_ip, get_pid, get_task_uid


@pytest.mark.unittest
class TestSystemHelper():

    def test_get(self):
        try:
            get_ip()
        except:
            pass
        assert isinstance(get_pid(), int)
        assert isinstance(get_task_uid(), str)
