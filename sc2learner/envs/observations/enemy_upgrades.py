'''
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. compute enemy upgrades from units infomation
'''
import torch
from collections import namedtuple
from sc2learner.torch_utils import one_hot
'''
UPGRADES_LIST = [
    "TerranInfantryWeaponsLevel1",
    "TerranInfantryWeaponsLevel2",
    "TerranInfantryWeaponsLevel3",
    "TerranInfantryArmorsLevel1",
    "TerranInfantryArmorsLevel2",
    "TerranInfantryArmorsLevel3",
    "TerranVehicleArmorsLevel1",
    "TerranVehicleArmorsLevel2",
    "TerranVehicleArmorsLevel3",
    "TerranVehicleWeaponsLevel1",
    "TerranVehicleWeaponsLevel2",
    "TerranVehicleWeaponsLevel3",
    "TerranShipWeaponsLevel1",
    "TerranShipWeaponsLevel2",
    "TerranShipWeaponsLevel3",
    "TerranShipArmorsLevel1",
    "TerranShipArmorsLevel2",
    "TerranShipArmorsLevel3",
    "ProtossGroundWeaponsLevel1",
    "ProtossGroundWeaponsLevel2",
    "ProtossGroundWeaponsLevel3",
    "ProtossGroundArmorsLevel1",
    "ProtossGroundArmorsLevel2",
    "ProtossGroundArmorsLevel3",
    "ProtossShieldsLevel1",
    "ProtossShieldsLevel2",
    "ProtossShieldsLevel3",
    "ProtossAirWeaponsLevel1",
    "ProtossAirWeaponsLevel2",
    "ProtossAirWeaponsLevel3",
    "ProtossAirArmorsLevel1",
    "ProtossAirArmorsLevel2",
    "ProtossAirArmorsLevel3",
    "ZergMeleeWeaponsLevel1",
    "ZergMeleeWeaponsLevel2",
    "ZergMeleeWeaponsLevel3",
    "ZergGroundArmorsLevel1",
    "ZergGroundArmorsLevel2",
    "ZergGroundArmorsLevel3",
    "ZergMissileWeaponsLevel1",
    "ZergMissileWeaponsLevel2",
    "ZergMissileWeaponsLevel3",
    "ZergFlyerWeaponsLevel1",
    "ZergFlyerWeaponsLevel2",
    "ZergFlyerWeaponsLevel3",
    "ZergFlyerArmorsLevel1",
    "ZergFlyerArmorsLevel2",
    "ZergFlyerArmorsLevel3",
]
'''

terran_infantry_unit_types = (50, 51, 48, 49)
terran_vehicle_unit_types = (692, 53, 484, 33, 32, 52, 691)
terran_ship_unit_types = (55, 689, 734, 34, 35, 57, 56)
protoss_ground_unit_types = (311, 141, 4, 83, 77, 74, 73)
protoss_air_unit_types = (79, 10, 78, 80, 496)
zerg_melee_unit_types = (9, 289, 109, 105)
zerg_missile_unit_types = (107, 126, 688, 110, 489, 502)
zerg_flyer_unit_types = (114, 112, 108)

UNIT2UPGRADE = namedtuple('UNIT2UPGRADE', ['unit_type_list', 'upgrade_idx'])
unit_type2upgrade_idx = [
    UNIT2UPGRADE(terran_infantry_unit_types, [0, 3]),
    UNIT2UPGRADE(terran_vehicle_unit_types, [6, 9]),
    UNIT2UPGRADE(terran_ship_unit_types, [12, 15]),
    UNIT2UPGRADE(protoss_ground_unit_types, [18, 21, 24]),
    UNIT2UPGRADE(protoss_air_unit_types, [27, 30, 24]),
    UNIT2UPGRADE(zerg_melee_unit_types, [33, 36]),
    UNIT2UPGRADE(zerg_missile_unit_types, [39, 36]),
    UNIT2UPGRADE(zerg_flyer_unit_types, [42, 45]),
]
shield_upgrade_idx = 24


def get_enemy_upgrades_raw_data(obs, upgrades=None):
    """
        Overview: get enemy upgrades according to the current obs and the last upgrades
        Arguments:
            - obs (:obj:`dict`): the dict of current obs(pysc2 obs)
            - upgrades (:obj:`torch.Tensor or None`): for the first frame, upgrades is None, otherwise
                upgrades is a shape([48]) tensor that is the last upgrades
        Returns:
            - upgrades (:obj:`torch.Tensor`): the current upgrades
    """
    if upgrades is None:
        upgrades = torch.zeros(48).long()
    units = obs['raw_units']
    for unit in units:
        if unit.alliance == 4:
            if unit.unit_type in terran_infantry_unit_types:
                if unit.attack_upgrade_level > 0:
                    level = unit.attack_upgrade_level
                    upgrades[-1 + level] = 1
                    if level > 1:
                        upgrades[-2 + level] = 0
                if unit.armor_upgrade_level > 0:
                    level = unit.armor_upgrade_level
                    upgrades[2 + level] = 1
                    if level > 1:
                        upgrades[1 + level] = 0
            elif unit.unit_type in terran_vehicle_unit_types:
                if unit.attack_upgrade_level > 0:
                    level = unit.attack_upgrade_level
                    upgrades[5 + level] = 1
                    if level > 1:
                        upgrades[4 + level] = 0
                if unit.armor_upgrade_level > 0:
                    level = unit.armor_upgrade_level
                    upgrades[8 + level] = 1
                    if level > 1:
                        upgrades[7 + level] = 0
            elif unit.unit_type in terran_ship_unit_types:
                if unit.attack_upgrade_level > 0:
                    level = unit.attack_upgrade_level
                    upgrades[11 + level] = 1
                    if level > 1:
                        upgrades[10 + level] = 0
                if unit.armor_upgrade_level > 0:
                    level = unit.armor_upgrade_level
                    upgrades[14 + level] = 1
                    if level > 1:
                        upgrades[13 + level] = 0
            elif unit.unit_type in protoss_ground_unit_types:
                if unit.attack_upgrade_level > 0:
                    level = unit.attack_upgrade_level
                    upgrades[17 + level] = 1
                    if level > 1:
                        upgrades[16 + level] = 0
                if unit.armor_upgrade_level > 0:
                    level = unit.armor_upgrade_level
                    upgrades[20 + level] = 1
                    if level > 1:
                        upgrades[19 + level] = 0
                if unit.shield_upgrade_level > 0:
                    level = unit.shield_upgrade_level
                    upgrades[23 + level] = 1
                    if level > 1:
                        upgrades[22 + level] = 0
            elif unit.unit_type in protoss_air_unit_types:
                if unit.attack_upgrade_level > 0:
                    level = unit.attack_upgrade_level
                    upgrades[26 + level] = 1
                    if level > 1:
                        upgrades[25 + level] = 0
                if unit.armor_upgrade_level > 0:
                    level = unit.armor_upgrade_level
                    upgrades[29 + level] = 1
                    if level > 1:
                        upgrades[28 + level] = 0
                if unit.shield_upgrade_level > 0:
                    level = unit.shield_upgrade_level
                    upgrades[23 + level] = 1
                    if level > 1:
                        upgrades[22 + level] = 0
            elif unit.unit_type in zerg_melee_unit_types:
                if unit.attack_upgrade_level > 0:
                    level = unit.attack_upgrade_level
                    upgrades[32 + level] = 1
                    if level > 1:
                        upgrades[31 + level] = 0
                if unit.armor_upgrade_level > 0:
                    level = unit.armor_upgrade_level
                    upgrades[35 + level] = 1
                    if level > 1:
                        upgrades[34 + level] = 0
            elif unit.unit_type in zerg_missile_unit_types:
                if unit.attack_upgrade_level > 0:
                    level = unit.attack_upgrade_level
                    upgrades[38 + level] = 1
                    if level > 1:
                        upgrades[37 + level] = 0
                if unit.armor_upgrade_level > 0:
                    level = unit.armor_upgrade_level
                    upgrades[35 + level] = 1
                    if level > 1:
                        upgrades[34 + level] = 0
            elif unit.unit_type in zerg_flyer_unit_types:
                if unit.attack_upgrade_level > 0:
                    level = unit.attack_upgrade_level
                    upgrades[41 + level] = 1
                    if level > 1:
                        upgrades[40 + level] = 0
                if unit.armor_upgrade_level > 0:
                    level = unit.armor_upgrade_level
                    upgrades[44 + level] = 1
                    if level > 1:
                        upgrades[43 + level] = 0
    return upgrades.float()


def get_enemy_upgrades_processed_data(obs, upgrades):
    """
        Overview: get enemy upgrades according to the current obs and the last upgrades
        Arguments:
            - obs (:obj:`dict`): the dict of current obs(pysc2 obs)
            - upgrades (:obj:`torch.Tensor or None`): for the first frame, upgrades is None, otherwise
                upgrades is a shape([48]) tensor that is the last upgrades
        Returns:
            - upgrades (:obj:`torch.Tensor`): the current upgrades
    """
    if upgrades is None:
        upgrades = torch.zeros(48).long()
    slices = {}
    slices['alliance'] = slice(263, 263 + 5)
    slices['attack_upgrade_level'] = slice(-16, -16 + 4)
    slices['armor_upgrade_level'] = slice(-12, -12 + 4)
    slices['shield_upgrade_level'] = slice(-8, -8 + 4)
    entity_info = obs['entity_info']
    assert entity_info.shape[1] == 1340, entity_info.shape  # the entity_info after the merge_action

    info = {k: [] for k in ['alliance', 'attack_upgrade_level', 'armor_upgrade_level', 'shield_upgrade_level']}
    for k in info.keys():
        info[k] = entity_info[:, slices[k]]
    # all the values are one-hot code
    for k in info.keys():
        info[k] = torch.max(info[k], dim=1)[1].tolist()
    # get enemy
    enemy_index = [idx for idx, v in enumerate(info['alliance']) if v == 4]
    # get the original unit type
    info['unit_type'] = obs['entity_raw']['type']
    level_map = {0: 'attack_upgrade_level', 1: 'armor_upgrade_level', 2: 'shield_upgrade_level'}

    for idx in enemy_index:
        unit_type = info['unit_type'][idx]
        type_idx = -1
        for t_idx, t in enumerate(unit_type2upgrade_idx):
            if unit_type in t.unit_type_list:
                type_idx = t_idx
        if type_idx != -1:
            upgrade_idx = unit_type2upgrade_idx[type_idx].upgrade_idx
            for i, u_idx in enumerate(upgrade_idx):
                val = info[level_map[i]][idx]
                if val > 0:
                    upgrades[u_idx:u_idx + 3] = one_hot(torch.LongTensor([val - 1]), 3)

    return upgrades.float()
