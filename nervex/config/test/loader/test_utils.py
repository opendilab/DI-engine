import pytest

from ...loader.utils import keep


@pytest.mark.unittest
class TestConfigLoaderUtils:
    def test_keep(self):
        _loader = keep()
        assert _loader(1) == 1
        assert _loader(2) == 2
        assert _loader(None) is None
