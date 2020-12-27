import pytest

from ...config import GLOBAL_HOST, LOCAL_HOST


@pytest.mark.unittest
class TestInteractionConfig:

    def test_base_host(self):
        assert GLOBAL_HOST == '0.0.0.0'
        assert LOCAL_HOST == '127.0.0.1'
