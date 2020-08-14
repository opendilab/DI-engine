import pytest
from ..action.alphastar_action import merge_same_id_action


@pytest.mark.unittest
def test_merge_same_id_action():
    # fake data, the same format
    actions = [
        {
            'action_type': [0],
            'selected_units': None,
            'target_units': None,
            'target_location': None
        },
        {
            'action_type': [0],
            'selected_units': None,
            'target_units': None,
            'target_location': None
        },
        {
            'action_type': [2],
            'selected_units': [112, 131],
            'target_units': [939],
            'target_location': None
        },
        {
            'action_type': [2],
            'selected_units': [132],
            'target_units': [939],
            'target_location': None
        },
        {
            'action_type': [2],
            'selected_units': [133],
            'target_units': [938],
            'target_location': None
        },
        {
            'action_type': [3],
            'selected_units': [132],
            'target_units': [939],
            'target_location': None
        },
        {
            'action_type': [4],
            'selected_units': [1321],
            'target_units': None,
            'target_location': None
        },
        {
            'action_type': [4],
            'selected_units': [1321, 1328],
            'target_units': None,
            'target_location': None
        },
        {
            'action_type': [5],
            'selected_units': [1321, 1328],
            'target_units': None,
            'target_location': [21, 43]
        },
        {
            'action_type': [5],
            'selected_units': [1322, 1327],
            'target_units': None,
            'target_location': [21, 43]
        },
        {
            'action_type': [5],
            'selected_units': [1323, 1326],
            'target_units': None,
            'target_location': [21, 42]
        },
    ]

    assert len(actions) == 11
    merged_actions = merge_same_id_action(actions)
    assert len(merged_actions) == 7
    for k in merged_actions:
        print(k)
