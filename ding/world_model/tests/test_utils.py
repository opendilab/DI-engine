import pytest
from ding.world_model.utils import get_rollout_length_scheduler


@pytest.mark.unittest
def test_get_rollout_length_scheduler():
    scheduler = get_rollout_length_scheduler()
    assert scheduler(0) == 1
    assert scheduler(19999) == 1
    assert scheduler(150000) == 25
    assert scheduler(1500000) == 25
