from pysc2.lib.action_dict import GENERAL_ACTION_INFO_MASK


# TODO The only usage of this function is in AS agent. Consider move this to there.
def get_location_mask(action_type, spatial_info):

    def has_zerg_unit(units):
        assert isinstance(units, list)
        for unit in units:
            if 'ZERG' in unit:
                return True
        return False

    assert len(spatial_info.shape) == 3
    creep = slice(7, 7 + 2)
    pathable = slice(16, 16 + 2)
    buildable = slice(18, 18 + 2)

    is_creep = spatial_info[creep][1:2]
    is_pathable = spatial_info[pathable][1:2]
    is_buildable = spatial_info[buildable][1:2]

    action_info = GENERAL_ACTION_INFO_MASK[action_type]
    if 'Build_' in action_info['name'] and action_info['target_location']:
        if action_info['name'] in ['Build_Hatchery_pt', 'Build_NydusWorm_pt']:
            return is_buildable
        elif has_zerg_unit(action_info['avail_unit_type']):
            return (is_creep.long() & is_buildable.long()).to(spatial_info.dtype)
        else:
            return is_buildable
    else:
        return is_pathable
