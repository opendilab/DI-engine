import pytest
import warnings
from ding.utils.deprecation import deprecated


@pytest.mark.unittest
def test_deprecated():

    @deprecated('0.4.1', '0.5.1')
    def deprecated_func1():
        pass

    @deprecated('0.4.1', '0.5.1', 'deprecated_func3')
    def deprecated_func2():
        pass

    with warnings.catch_warnings(record=True) as w:
        deprecated_func1()
        assert (
            'API `test_deprecation.deprecated_func1` is deprecated '
            'since version 0.4.1 and will be removed in version 0.5.1.'
        ) == str(w[-1].message)
        deprecated_func2()
        assert (
            'API `test_deprecation.deprecated_func2` is deprecated '
            'since version 0.4.1 and will be removed in version 0.5.1, '
            'please use `deprecated_func3` instead.'
        ) == str(w[-1].message)
