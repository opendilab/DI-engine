import py
import pytest
from ding.framework import EventEnum


@pytest.mark.unittest
def test_event_enum():
    assert EventEnum.TEMPLATE_EVENT.get_event(12, 34) == "example_xxx_12_34"
    assert EventEnum.TEMPLATE_EVENT.get_event(12, player_id=34) == "example_xxx_12_34"
    assert EventEnum.TEMPLATE_EVENT.get_event(actor_id=12, player_id=34) == "example_xxx_12_34"