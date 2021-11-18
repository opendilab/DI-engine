import pytest
from ding.utils.registry import Registry


@pytest.mark.unittest
def test_registry():
    TEST_REGISTRY = Registry()

    @TEST_REGISTRY.register('a')
    class A:
        pass

    instance = TEST_REGISTRY.build('a')
    assert isinstance(instance, A)

    with pytest.raises(AssertionError):

        @TEST_REGISTRY.register('a')
        class A1:
            pass

    @TEST_REGISTRY.register('a', force_overwrite=True)
    class A2:
        pass

    instance = TEST_REGISTRY.build('a')
    assert isinstance(instance, A2)
