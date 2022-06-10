import pytest
from easydict import EasyDict
from ding.world_model.utils import get_rollout_length_scheduler


@pytest.mark.unittest
def test_get_rollout_length_scheduler():
    fake_cfg = EasyDict(
        type='linear',
        rollout_start_step=20000,
        rollout_end_step=150000,
        rollout_length_min=1,
        rollout_length_max=25,
    )
    scheduler = get_rollout_length_scheduler(fake_cfg)
    assert scheduler(0) == 1
    assert scheduler(19999) == 1
    assert scheduler(150000) == 25
    assert scheduler(1500000) == 25
