import os
import sys
import ceph
import torch
import numpy as np
import pickle

from sc2learner.utils import read_file_ceph
from pysc2.lib.units import Neutral, Protoss, Terran, Zerg
from pysc2.lib.action_dict import ACTION_INFO_MASK, ACTIONS_STAT

result = {}


def save_obj(obj, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(load_path):
    with open(load_path, 'rb') as f:
        return pickle.load(f)


def analyse_stat(stat_path):
    try:
        stat = read_file_ceph(stat_path.strip(), read_type='pickle')
        # stat = torch.load(stat)
        if 'action_statistics' in stat:
            for action_type, statistics in stat['action_statistics'].items():
                if action_type not in result:
                    result[action_type] = [set(), set()]
                if statistics['selected_type']:
                    result[action_type][0] = result[action_type][0] | statistics['selected_type']
                if statistics['target_type']:
                    result[action_type][1] = result[action_type][1] | statistics['target_type']
        return True
    except FileNotFoundError as e:
        print('failed to read {}'.format(stat_path))
        return False


def main():
    stat_list_path = '/mnt/lustre/zhangming/data/replay_decode_48.upload.list'
    ps = open(stat_list_path, 'r').readlines()

    for i, p in enumerate(ps):
        p = p.strip()
        if p.endswith('stat'):
            print(p.strip())
            analyse_stat(p)
            # print(result)

    print('----------------------------------')
    print(result)

    save_obj(result, '/mnt/lustre/zhangming/data/stat_info.pkl')


def merge_units():
    ret = {}
    for x in [Neutral, Protoss, Terran, Zerg]:
        for item in x:
            assert item.value not in ret, '{} {} / {}'.format(item.value, item.name, ret[item.value])
            ret[item.value] = item.name
    return ret


if __name__ == '__main__':
    # main()

    ACTIONS_STAT_NEW = {
        0: {
            'action_name': 'no_op',
            'selected_type': [],
            'selected_type_name': [],
            'target_type': [],
            'target_type_name': []
        },
        1: {
            'action_name': 'Smart_pt',
            'selected_type': [
                4, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 21, 24, 27, 28, 33, 34, 35, 36, 43, 44, 45, 46, 48, 49, 50,
                51, 52, 53, 54, 55, 56, 57, 59, 62, 67, 71, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 86, 95, 100,
                101, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 118, 126, 127, 128, 129, 130, 132, 134,
                139, 140, 141, 142, 144, 145, 150, 151, 687, 688, 689, 691, 692, 693, 694, 733, 268, 801, 289, 311,
                1912, 892, 893, 484, 489, 494, 495, 496, 498, 499, 501, 502
            ],
            'selected_type_name': [
                'Colossus', 'InfestedTerran', 'BanelingCocoon', 'Baneling', 'Mothership', 'Changeling',
                'ChangelingZealot', 'ChangelingMarineShield', 'ChangelingMarine', 'ChangelingZerglingWings',
                'ChangelingZergling', 'CommandCenter', 'Barracks', 'Bunker', 'Factory', 'Starport', 'SiegeTank',
                'VikingAssault', 'VikingFighter', 'CommandCenterFlying', 'FactoryFlying', 'StarportFlying', 'SCV',
                'BarracksFlying', 'Marine', 'Reaper', 'Ghost', 'Marauder', 'Thor', 'Hellion', 'Medivac', 'Banshee',
                'Raven', 'Battlecruiser', 'Nexus', 'Gateway', 'Stargate', 'RoboticsFacility', 'Zealot', 'Stalker',
                'HighTemplar', 'DarkTemplar', 'Sentry', 'Phoenix', 'Carrier', 'VoidRay', 'WarpPrism', 'Observer',
                'Immortal', 'Probe', 'Hatchery', 'NydusNetwork', 'Lair', 'Hive', 'Cocoon', 'Drone', 'Zergling',
                'Overlord', 'Hydralisk', 'Mutalisk', 'Ultralisk', 'Roach', 'Infestor', 'Corruptor', 'BroodLordCocoon',
                'BroodLord', 'RoachBurrowed', 'Queen', 'InfestorBurrowed', 'OverseerCocoon', 'Overseer',
                'PlanetaryFortress', 'OrbitalCommand', 'OrbitalCommandFlying', 'SpineCrawlerUprooted',
                'SporeCrawlerUprooted', 'Archon', 'NydusCanal', 'GhostAlternate', 'GhostNova', 'InfestedTerranCocoon',
                'Larva', 'RavagerCocoon', 'Ravager', 'Liberator', 'ThorHighImpactMode', 'Cyclone', 'LocustFlying',
                'Disruptor', 'DisruptorPhased', 'MULE', 'AdeptPhaseShift', 'Broodling', 'Adept',
                'OverseerOversightMode', 'OverlordTransportCocoon', 'OverlordTransport', 'Hellbat', 'Locust',
                'SwarmHost', 'Oracle', 'Tempest', 'WidowMine', 'Viper', 'LurkerCocoon', 'Lurker'
            ],
            'target_type': [],
            'target_type_name': []
        },
        2: {
            'action_name': 'Attack_pt',
            'selected_type': [
                128, 129, 4, 134, 7, 688, 9, 10, 139, 268, 13, 12, 14, 15, 16, 17, 140, 141, 144, 145, 150, 489, 32,
                801, 33, 35, 34, 289, 45, 46, 48, 49, 50, 51, 52, 53, 54, 311, 56, 57, 55, 689, 691, 692, 693, 694, 73,
                74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 733, 734, 484, 104, 105, 106, 107, 108, 109, 110, 111, 112,
                113, 114, 494, 495, 496, 118, 498, 499, 502, 892, 893, 126, 127
            ],
            'selected_type_name': [
                'OverseerCocoon', 'Overseer', 'Colossus', 'OrbitalCommandFlying', 'InfestedTerran', 'Ravager',
                'Baneling', 'Mothership', 'SpineCrawlerUprooted', 'MULE', 'ChangelingZealot', 'Changeling',
                'ChangelingMarineShield', 'ChangelingMarine', 'ChangelingZerglingWings', 'ChangelingZergling',
                'SporeCrawlerUprooted', 'Archon', 'GhostAlternate', 'GhostNova', 'InfestedTerranCocoon', 'Locust',
                'SiegeTankSieged', 'AdeptPhaseShift', 'SiegeTank', 'VikingFighter', 'VikingAssault', 'Broodling', 'SCV',
                'BarracksFlying', 'Marine', 'Reaper', 'Ghost', 'Marauder', 'Thor', 'Hellion', 'Medivac', 'Adept',
                'Raven', 'Battlecruiser', 'Banshee', 'Liberator', 'ThorHighImpactMode', 'Cyclone', 'LocustFlying',
                'Disruptor', 'Zealot', 'Stalker', 'HighTemplar', 'DarkTemplar', 'Sentry', 'Phoenix', 'Carrier',
                'VoidRay', 'WarpPrism', 'Observer', 'Immortal', 'Probe', 'DisruptorPhased', 'LiberatorAG', 'Hellbat',
                'Drone', 'Zergling', 'Overlord', 'Hydralisk', 'Mutalisk', 'Ultralisk', 'Roach', 'Infestor', 'Corruptor',
                'BroodLordCocoon', 'BroodLord', 'SwarmHost', 'Oracle', 'Tempest', 'RoachBurrowed', 'WidowMine', 'Viper',
                'Lurker', 'OverlordTransportCocoon', 'OverlordTransport', 'Queen', 'InfestorBurrowed'
            ],
            'target_type': [],
            'target_type_name': []
        },
        3: {
            'action_name': 'Attack_unit',
            'selected_type': [
                4, 7, 9, 10, 12, 13, 14, 15, 16, 17, 24, 31, 32, 33, 34, 35, 36, 45, 46, 48, 49, 50, 51, 52, 53, 54, 55,
                56, 57, 66, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 98, 99, 104, 105, 106, 107, 108, 109, 110,
                111, 112, 113, 114, 118, 126, 127, 128, 129, 130, 139, 140, 141, 144, 150, 688, 689, 691, 692, 693, 694,
                733, 734, 268, 801, 289, 311, 1912, 893, 484, 489, 494, 495, 496, 498, 499, 502, 503
            ],
            'selected_type_name': [
                'Colossus', 'InfestedTerran', 'Baneling', 'Mothership', 'Changeling', 'ChangelingZealot',
                'ChangelingMarineShield', 'ChangelingMarine', 'ChangelingZerglingWings', 'ChangelingZergling', 'Bunker',
                'AutoTurret', 'SiegeTankSieged', 'SiegeTank', 'VikingAssault', 'VikingFighter', 'CommandCenterFlying',
                'SCV', 'BarracksFlying', 'Marine', 'Reaper', 'Ghost', 'Marauder', 'Thor', 'Hellion', 'Medivac',
                'Banshee', 'Raven', 'Battlecruiser', 'PhotonCannon', 'Zealot', 'Stalker', 'HighTemplar', 'DarkTemplar',
                'Sentry', 'Phoenix', 'Carrier', 'VoidRay', 'WarpPrism', 'Observer', 'Immortal', 'Probe', 'SpineCrawler',
                'SporeCrawler', 'Drone', 'Zergling', 'Overlord', 'Hydralisk', 'Mutalisk', 'Ultralisk', 'Roach',
                'Infestor', 'Corruptor', 'BroodLordCocoon', 'BroodLord', 'RoachBurrowed', 'Queen', 'InfestorBurrowed',
                'OverseerCocoon', 'Overseer', 'PlanetaryFortress', 'SpineCrawlerUprooted', 'SporeCrawlerUprooted',
                'Archon', 'GhostAlternate', 'InfestedTerranCocoon', 'Ravager', 'Liberator', 'ThorHighImpactMode',
                'Cyclone', 'LocustFlying', 'Disruptor', 'DisruptorPhased', 'LiberatorAG', 'MULE', 'AdeptPhaseShift',
                'Broodling', 'Adept', 'OverseerOversightMode', 'OverlordTransport', 'Hellbat', 'Locust', 'SwarmHost',
                'Oracle', 'Tempest', 'WidowMine', 'Viper', 'Lurker', 'LurkerBurrowed'
            ],
            'target_type': [
                4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 589, 78, 77, 80, 79, 82, 83, 84,
                81, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 612, 102, 101, 103, 104, 105, 106, 107,
                108, 110, 111, 112, 109, 114, 113, 116, 628, 118, 630, 117, 119, 115, 125, 126, 639, 128, 641, 129, 127,
                132, 133, 134, 130, 136, 137, 138, 139, 140, 141, 142, 144, 131, 661, 150, 151, 149, 665, 666, 687, 688,
                689, 690, 691, 692, 693, 694, 732, 734, 268, 289, 801, 311, 341, 342, 343, 364, 365, 880, 884, 373,
                1910, 1911, 1912, 885, 892, 893, 472, 474, 475, 483, 484, 485, 489, 493, 494, 495, 496, 498, 499, 500,
                501, 502, 503, 504
            ],
            'target_type_name': [
                'Colossus', 'TechLab', 'Reactor', 'InfestedTerran', 'BanelingCocoon', 'Baneling', 'Mothership',
                'Changeling', 'ChangelingZealot', 'ChangelingMarineShield', 'ChangelingMarine',
                'ChangelingZerglingWings', 'ChangelingZergling', 'CommandCenter', 'SupplyDepot', 'Refinery', 'Barracks',
                'EngineeringBay', 'MissileTurret', 'Bunker', 'SensorTower', 'GhostAcademy', 'Factory', 'Starport',
                'Armory', 'FusionCore', 'AutoTurret', 'SiegeTankSieged', 'SiegeTank', 'VikingAssault', 'VikingFighter',
                'CommandCenterFlying', 'BarracksTechLab', 'BarracksReactor', 'FactoryTechLab', 'FactoryReactor',
                'StarportTechLab', 'StarportReactor', 'FactoryFlying', 'StarportFlying', 'SCV', 'BarracksFlying',
                'SupplyDepotLowered', 'Marine', 'Reaper', 'Ghost', 'Marauder', 'Thor', 'Hellion', 'Medivac', 'Banshee',
                'Raven', 'Battlecruiser', 'Nexus', 'Pylon', 'Assimilator', 'Gateway', 'Forge', 'FleetBeacon',
                'TwilightCouncil', 'PhotonCannon', 'Stargate', 'TemplarArchive', 'DarkShrine', 'RoboticsBay',
                'RoboticsFacility', 'CyberneticsCore', 'Zealot', 'Stalker', 'HighTemplar', 'DarkTemplar',
                'CollapsibleTerranTowerDiagonal', 'Phoenix', 'Sentry', 'VoidRay', 'Carrier', 'Observer', 'Immortal',
                'Probe', 'WarpPrism', 'Hatchery', 'CreepTumor', 'Extractor', 'SpawningPool', 'EvolutionChamber',
                'HydraliskDen', 'Spire', 'UltraliskCavern', 'InfestationPit', 'NydusNetwork', 'BanelingNest',
                'RoachWarren', 'SpineCrawler', 'SporeCrawler', 'Lair', 'CleaningBot', 'GreaterSpire', 'Hive', 'Cocoon',
                'Drone', 'Zergling', 'Overlord', 'Hydralisk', 'Mutalisk', 'Roach', 'Infestor', 'Corruptor', 'Ultralisk',
                'BroodLord', 'BroodLordCocoon', 'DroneBurrowed', 'DestructibleCityDebris4x4', 'RoachBurrowed',
                'DestructibleCityDebrisHugeDiagonalBLUR', 'HydraliskBurrowed', 'ZerglingBurrowed', 'BanelingBurrowed',
                'QueenBurrowed', 'Queen', 'DestructibleRockEx16x6', 'OverseerCocoon',
                'DestructibleRockEx1DiagonalHugeBLUR', 'Overseer', 'InfestorBurrowed', 'OrbitalCommand', 'WarpGate',
                'OrbitalCommandFlying', 'PlanetaryFortress', 'WarpPrismPhasing', 'CreepTumorBurrowed',
                'CreepTumorQueen', 'SpineCrawlerUprooted', 'SporeCrawlerUprooted', 'Archon', 'NydusCanal',
                'GhostAlternate', 'UltraliskBurrowed', 'LabBot', 'InfestedTerranCocoon', 'Larva', 'XelNagaTower',
                'LabMineralField', 'LabMineralField750', 'RavagerCocoon', 'Ravager', 'Liberator', 'RavagerBurrowed',
                'ThorHighImpactMode', 'Cyclone', 'LocustFlying', 'Disruptor', 'StasisTrap', 'LiberatorAG', 'MULE',
                'Broodling', 'AdeptPhaseShift', 'Adept', 'MineralField', 'VespeneGeyser', 'SpacePlatformGeyser',
                'DestructibleDebris4x4', 'DestructibleDebris6x6', 'PurifierVespeneGeyser', 'PurifierMineralField',
                'DestructibleRampDiagonalHugeBLUR', 'ShieldBattery', 'ObserverSurveillanceMode',
                'OverseerOversightMode', 'PurifierMineralField750', 'OverlordTransportCocoon', 'OverlordTransport',
                'UnbuildableRocksDestructible', 'UnbuildablePlatesDestructible', 'Debris2x2NonConjoined',
                'MineralField750', 'Hellbat', 'CollapsibleTerranTowerDebris', 'Locust', 'SwarmHostBurrowed',
                'SwarmHost', 'Oracle', 'Tempest', 'WidowMine', 'Viper', 'WidowMineBurrowed', 'LurkerCocoon', 'Lurker',
                'LurkerBurrowed', 'LurkerDen'
            ]
        },
        12: {
            'action_name': 'Smart_unit',
            'selected_type': [
                4, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 21, 23, 24, 27, 28, 31, 32, 33, 34, 35, 36, 43, 44, 45, 46,
                48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 62, 66, 67, 71, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
                84, 86, 95, 98, 99, 100, 101, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 118, 126, 127,
                128, 129, 130, 132, 134, 136, 139, 140, 141, 142, 144, 150, 151, 687, 688, 689, 691, 692, 693, 694, 733,
                734, 268, 801, 289, 311, 1910, 1912, 892, 893, 484, 489, 494, 495, 496, 498, 499, 500, 501, 502, 503
            ],
            'selected_type_name': [
                'Colossus', 'InfestedTerran', 'BanelingCocoon', 'Baneling', 'Mothership', 'Changeling',
                'ChangelingZealot', 'ChangelingMarineShield', 'ChangelingMarine', 'ChangelingZerglingWings',
                'ChangelingZergling', 'CommandCenter', 'Barracks', 'MissileTurret', 'Bunker', 'Factory', 'Starport',
                'AutoTurret', 'SiegeTankSieged', 'SiegeTank', 'VikingAssault', 'VikingFighter', 'CommandCenterFlying',
                'FactoryFlying', 'StarportFlying', 'SCV', 'BarracksFlying', 'Marine', 'Reaper', 'Ghost', 'Marauder',
                'Thor', 'Hellion', 'Medivac', 'Banshee', 'Raven', 'Battlecruiser', 'Nexus', 'Gateway', 'PhotonCannon',
                'Stargate', 'RoboticsFacility', 'Zealot', 'Stalker', 'HighTemplar', 'DarkTemplar', 'Sentry', 'Phoenix',
                'Carrier', 'VoidRay', 'WarpPrism', 'Observer', 'Immortal', 'Probe', 'Hatchery', 'NydusNetwork',
                'SpineCrawler', 'SporeCrawler', 'Lair', 'Hive', 'Cocoon', 'Drone', 'Zergling', 'Overlord', 'Hydralisk',
                'Mutalisk', 'Ultralisk', 'Roach', 'Infestor', 'Corruptor', 'BroodLordCocoon', 'BroodLord',
                'RoachBurrowed', 'Queen', 'InfestorBurrowed', 'OverseerCocoon', 'Overseer', 'PlanetaryFortress',
                'OrbitalCommand', 'OrbitalCommandFlying', 'WarpPrismPhasing', 'SpineCrawlerUprooted',
                'SporeCrawlerUprooted', 'Archon', 'NydusCanal', 'GhostAlternate', 'InfestedTerranCocoon', 'Larva',
                'RavagerCocoon', 'Ravager', 'Liberator', 'ThorHighImpactMode', 'Cyclone', 'LocustFlying', 'Disruptor',
                'DisruptorPhased', 'LiberatorAG', 'MULE', 'AdeptPhaseShift', 'Broodling', 'Adept', 'ShieldBattery',
                'OverseerOversightMode', 'OverlordTransportCocoon', 'OverlordTransport', 'Hellbat', 'Locust',
                'SwarmHost', 'Oracle', 'Tempest', 'WidowMine', 'Viper', 'WidowMineBurrowed', 'LurkerCocoon', 'Lurker',
                'LurkerBurrowed'
            ],
            'target_type': [
                4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
                589, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 612, 102, 101, 103, 104, 105, 106,
                107, 109, 110, 111, 112, 113, 108, 114, 628, 117, 630, 115, 118, 119, 116, 125, 126, 639, 127, 129, 641,
                128, 132, 133, 130, 134, 136, 137, 138, 139, 140, 141, 142, 144, 145, 146, 147, 131, 149, 661, 151, 150,
                665, 666, 687, 688, 689, 690, 691, 692, 693, 694, 732, 733, 734, 268, 796, 797, 289, 801, 311, 341, 342,
                343, 364, 365, 880, 881, 884, 885, 1910, 373, 1912, 1911, 892, 893, 85, 472, 474, 475, 483, 484, 485,
                489, 493, 494, 495, 496, 498, 499, 500, 501, 502, 503, 504
            ],
            'target_type_name': [
                'Colossus', 'TechLab', 'Reactor', 'InfestedTerran', 'BanelingCocoon', 'Baneling', 'Mothership',
                'Changeling', 'ChangelingZealot', 'ChangelingMarineShield', 'ChangelingMarine',
                'ChangelingZerglingWings', 'ChangelingZergling', 'CommandCenter', 'SupplyDepot', 'Refinery', 'Barracks',
                'EngineeringBay', 'MissileTurret', 'Bunker', 'SensorTower', 'GhostAcademy', 'Factory', 'Starport',
                'Armory', 'FusionCore', 'AutoTurret', 'SiegeTankSieged', 'SiegeTank', 'VikingAssault', 'VikingFighter',
                'CommandCenterFlying', 'BarracksTechLab', 'BarracksReactor', 'FactoryTechLab', 'FactoryReactor',
                'StarportTechLab', 'StarportReactor', 'FactoryFlying', 'StarportFlying', 'SCV', 'BarracksFlying',
                'SupplyDepotLowered', 'Marine', 'Reaper', 'Ghost', 'Marauder', 'Thor', 'Hellion', 'Medivac', 'Banshee',
                'Raven', 'Battlecruiser', 'Nexus', 'Pylon', 'Assimilator', 'Gateway', 'Forge', 'FleetBeacon',
                'TwilightCouncil', 'PhotonCannon', 'Stargate', 'TemplarArchive', 'DarkShrine', 'RoboticsBay',
                'RoboticsFacility', 'CyberneticsCore', 'Zealot', 'Stalker', 'HighTemplar', 'DarkTemplar', 'Sentry',
                'Phoenix', 'Carrier', 'VoidRay', 'WarpPrism', 'Observer', 'Immortal', 'Probe',
                'CollapsibleTerranTowerDiagonal', 'Hatchery', 'CreepTumor', 'Extractor', 'SpawningPool',
                'EvolutionChamber', 'HydraliskDen', 'Spire', 'UltraliskCavern', 'InfestationPit', 'NydusNetwork',
                'BanelingNest', 'RoachWarren', 'SpineCrawler', 'SporeCrawler', 'Lair', 'CleaningBot', 'GreaterSpire',
                'Hive', 'Cocoon', 'Drone', 'Zergling', 'Overlord', 'Hydralisk', 'Ultralisk', 'Roach', 'Infestor',
                'Corruptor', 'BroodLordCocoon', 'Mutalisk', 'BroodLord', 'DestructibleCityDebris4x4',
                'HydraliskBurrowed', 'DestructibleCityDebrisHugeDiagonalBLUR', 'BanelingBurrowed', 'RoachBurrowed',
                'ZerglingBurrowed', 'DroneBurrowed', 'QueenBurrowed', 'Queen', 'DestructibleRockEx16x6',
                'InfestorBurrowed', 'Overseer', 'DestructibleRockEx1DiagonalHugeBLUR', 'OverseerCocoon',
                'OrbitalCommand', 'WarpGate', 'PlanetaryFortress', 'OrbitalCommandFlying', 'WarpPrismPhasing',
                'CreepTumorBurrowed', 'CreepTumorQueen', 'SpineCrawlerUprooted', 'SporeCrawlerUprooted', 'Archon',
                'NydusCanal', 'GhostAlternate', 'GhostNova', 'RichMineralField', 'RichMineralField750',
                'UltraliskBurrowed', 'XelNagaTower', 'LabBot', 'Larva', 'InfestedTerranCocoon', 'LabMineralField',
                'LabMineralField750', 'RavagerCocoon', 'Ravager', 'Liberator', 'RavagerBurrowed', 'ThorHighImpactMode',
                'Cyclone', 'LocustFlying', 'Disruptor', 'StasisTrap', 'DisruptorPhased', 'LiberatorAG', 'MULE',
                'PurifierRichMineralField', 'PurifierRichMineralField750', 'Broodling', 'AdeptPhaseShift', 'Adept',
                'MineralField', 'VespeneGeyser', 'SpacePlatformGeyser', 'DestructibleDebris4x4',
                'DestructibleDebris6x6', 'PurifierVespeneGeyser', 'ShakurasVespeneGeyser', 'PurifierMineralField',
                'PurifierMineralField750', 'ShieldBattery', 'DestructibleRampDiagonalHugeBLUR', 'OverseerOversightMode',
                'ObserverSurveillanceMode', 'OverlordTransportCocoon', 'OverlordTransport', 'Interceptor',
                'UnbuildableRocksDestructible', 'UnbuildablePlatesDestructible', 'Debris2x2NonConjoined',
                'MineralField750', 'Hellbat', 'CollapsibleTerranTowerDebris', 'Locust', 'SwarmHostBurrowed',
                'SwarmHost', 'Oracle', 'Tempest', 'WidowMine', 'Viper', 'WidowMineBurrowed', 'LurkerCocoon', 'Lurker',
                'LurkerBurrowed', 'LurkerDen'
            ]
        },
        13: {
            'action_name': 'Move_pt',
            'selected_type': [
                128, 129, 4, 134, 9, 689, 12, 13, 268, 15, 14, 16, 17, 140, 141, 33, 34, 35, 36, 289, 43, 44, 45, 46,
                48, 49, 688, 51, 52, 53, 54, 55, 311, 57, 56, 691, 692, 693, 73, 74, 75, 76, 77, 78, 80, 81, 82, 83, 84,
                484, 104, 105, 106, 107, 108, 109, 110, 489, 112, 494, 114, 495, 498, 499, 118, 502, 893, 126, 127
            ],
            'selected_type_name': [
                'OverseerCocoon', 'Overseer', 'Colossus', 'OrbitalCommandFlying', 'Baneling', 'Liberator', 'Changeling',
                'ChangelingZealot', 'MULE', 'ChangelingMarine', 'ChangelingMarineShield', 'ChangelingZerglingWings',
                'ChangelingZergling', 'SporeCrawlerUprooted', 'Archon', 'SiegeTank', 'VikingAssault', 'VikingFighter',
                'CommandCenterFlying', 'Broodling', 'FactoryFlying', 'StarportFlying', 'SCV', 'BarracksFlying',
                'Marine', 'Reaper', 'Ravager', 'Marauder', 'Thor', 'Hellion', 'Medivac', 'Banshee', 'Adept',
                'Battlecruiser', 'Raven', 'ThorHighImpactMode', 'Cyclone', 'LocustFlying', 'Zealot', 'Stalker',
                'HighTemplar', 'DarkTemplar', 'Sentry', 'Phoenix', 'VoidRay', 'WarpPrism', 'Observer', 'Immortal',
                'Probe', 'Hellbat', 'Drone', 'Zergling', 'Overlord', 'Hydralisk', 'Mutalisk', 'Ultralisk', 'Roach',
                'Locust', 'Corruptor', 'SwarmHost', 'BroodLord', 'Oracle', 'WidowMine', 'Viper', 'RoachBurrowed',
                'Lurker', 'OverlordTransport', 'Queen', 'InfestorBurrowed'
            ],
            'target_type': [],
            'target_type_name': []
        },
        14: {
            'action_name': 'Move_unit',
            'selected_type': [
                128, 129, 4, 134, 688, 9, 10, 139, 12, 13, 14, 15, 16, 17, 140, 689, 801, 33, 34, 35, 43, 45, 46, 48,
                49, 50, 51, 52, 53, 54, 55, 56, 311, 57, 691, 692, 694, 73, 74, 75, 77, 78, 79, 80, 81, 82, 83, 84, 484,
                104, 105, 106, 107, 108, 489, 110, 495, 112, 113, 114, 496, 498, 499, 118, 502, 126, 127
            ],
            'selected_type_name': [
                'OverseerCocoon', 'Overseer', 'Colossus', 'OrbitalCommandFlying', 'Ravager', 'Baneling', 'Mothership',
                'SpineCrawlerUprooted', 'Changeling', 'ChangelingZealot', 'ChangelingMarineShield', 'ChangelingMarine',
                'ChangelingZerglingWings', 'ChangelingZergling', 'SporeCrawlerUprooted', 'Liberator', 'AdeptPhaseShift',
                'SiegeTank', 'VikingAssault', 'VikingFighter', 'FactoryFlying', 'SCV', 'BarracksFlying', 'Marine',
                'Reaper', 'Ghost', 'Marauder', 'Thor', 'Hellion', 'Medivac', 'Banshee', 'Raven', 'Adept',
                'Battlecruiser', 'ThorHighImpactMode', 'Cyclone', 'Disruptor', 'Zealot', 'Stalker', 'HighTemplar',
                'Sentry', 'Phoenix', 'Carrier', 'VoidRay', 'WarpPrism', 'Observer', 'Immortal', 'Probe', 'Hellbat',
                'Drone', 'Zergling', 'Overlord', 'Hydralisk', 'Mutalisk', 'Locust', 'Roach', 'Oracle', 'Corruptor',
                'BroodLordCocoon', 'BroodLord', 'Tempest', 'WidowMine', 'Viper', 'RoachBurrowed', 'Lurker', 'Queen',
                'InfestorBurrowed'
            ],
            'target_type': [
                4, 9, 10, 18, 19, 20, 21, 22, 23, 24, 27, 28, 29, 32, 33, 34, 35, 36, 37, 45, 46, 47, 48, 49, 50, 51,
                52, 53, 54, 55, 57, 59, 60, 61, 62, 63, 65, 66, 67, 68, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 589, 83,
                84, 86, 88, 89, 90, 91, 92, 96, 97, 98, 99, 100, 101, 103, 104, 105, 106, 107, 108, 109, 110, 112, 114,
                126, 127, 639, 129, 132, 133, 134, 136, 137, 141, 149, 665, 666, 687, 688, 689, 691, 692, 268, 801, 311,
                341, 342, 343, 880, 884, 885, 1910, 373, 893, 472, 474, 483, 484, 494, 495, 496, 498, 502, 503
            ],
            'target_type_name': [
                'Colossus', 'Baneling', 'Mothership', 'CommandCenter', 'SupplyDepot', 'Refinery', 'Barracks',
                'EngineeringBay', 'MissileTurret', 'Bunker', 'Factory', 'Starport', 'Armory', 'SiegeTankSieged',
                'SiegeTank', 'VikingAssault', 'VikingFighter', 'CommandCenterFlying', 'BarracksTechLab', 'SCV',
                'BarracksFlying', 'SupplyDepotLowered', 'Marine', 'Reaper', 'Ghost', 'Marauder', 'Thor', 'Hellion',
                'Medivac', 'Banshee', 'Battlecruiser', 'Nexus', 'Pylon', 'Assimilator', 'Gateway', 'Forge',
                'TwilightCouncil', 'PhotonCannon', 'Stargate', 'TemplarArchive', 'CyberneticsCore', 'Zealot', 'Stalker',
                'HighTemplar', 'DarkTemplar', 'Sentry', 'Phoenix', 'Carrier', 'VoidRay', 'WarpPrism',
                'CollapsibleTerranTowerDiagonal', 'Immortal', 'Probe', 'Hatchery', 'Extractor', 'SpawningPool',
                'EvolutionChamber', 'HydraliskDen', 'Spire', 'BanelingNest', 'RoachWarren', 'SpineCrawler',
                'SporeCrawler', 'Lair', 'Hive', 'Cocoon', 'Drone', 'Zergling', 'Overlord', 'Hydralisk', 'Mutalisk',
                'Ultralisk', 'Roach', 'Corruptor', 'BroodLord', 'Queen', 'InfestorBurrowed', 'DestructibleRockEx16x6',
                'Overseer', 'OrbitalCommand', 'WarpGate', 'OrbitalCommandFlying', 'WarpPrismPhasing',
                'CreepTumorBurrowed', 'Archon', 'XelNagaTower', 'LabMineralField', 'LabMineralField750',
                'RavagerCocoon', 'Ravager', 'Liberator', 'ThorHighImpactMode', 'Cyclone', 'MULE', 'AdeptPhaseShift',
                'Adept', 'MineralField', 'VespeneGeyser', 'SpacePlatformGeyser', 'PurifierVespeneGeyser',
                'PurifierMineralField', 'PurifierMineralField750', 'ShieldBattery', 'DestructibleRampDiagonalHugeBLUR',
                'OverlordTransport', 'UnbuildableRocksDestructible', 'UnbuildablePlatesDestructible', 'MineralField750',
                'Hellbat', 'SwarmHost', 'Oracle', 'Tempest', 'WidowMine', 'Lurker', 'LurkerBurrowed'
            ]
        },
        15: {
            'action_name': 'Patrol_pt',
            'selected_type': [
                128, 129, 4, 134, 7, 688, 9, 10, 139, 268, 12, 13, 15, 14, 16, 17, 141, 144, 150, 489, 801, 34, 35, 33,
                36, 289, 43, 44, 45, 46, 48, 49, 50, 51, 52, 53, 54, 55, 56, 311, 57, 689, 691, 692, 693, 73, 74, 75,
                76, 77, 78, 79, 80, 81, 82, 83, 84, 484, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 494,
                495, 496, 118, 498, 499, 502, 893, 126, 127
            ],
            'selected_type_name': [
                'OverseerCocoon', 'Overseer', 'Colossus', 'OrbitalCommandFlying', 'InfestedTerran', 'Ravager',
                'Baneling', 'Mothership', 'SpineCrawlerUprooted', 'MULE', 'Changeling', 'ChangelingZealot',
                'ChangelingMarine', 'ChangelingMarineShield', 'ChangelingZerglingWings', 'ChangelingZergling', 'Archon',
                'GhostAlternate', 'InfestedTerranCocoon', 'Locust', 'AdeptPhaseShift', 'VikingAssault', 'VikingFighter',
                'SiegeTank', 'CommandCenterFlying', 'Broodling', 'FactoryFlying', 'StarportFlying', 'SCV',
                'BarracksFlying', 'Marine', 'Reaper', 'Ghost', 'Marauder', 'Thor', 'Hellion', 'Medivac', 'Banshee',
                'Raven', 'Adept', 'Battlecruiser', 'Liberator', 'ThorHighImpactMode', 'Cyclone', 'LocustFlying',
                'Zealot', 'Stalker', 'HighTemplar', 'DarkTemplar', 'Sentry', 'Phoenix', 'Carrier', 'VoidRay',
                'WarpPrism', 'Observer', 'Immortal', 'Probe', 'Hellbat', 'Drone', 'Zergling', 'Overlord', 'Hydralisk',
                'Mutalisk', 'Ultralisk', 'Roach', 'Infestor', 'Corruptor', 'BroodLordCocoon', 'BroodLord', 'SwarmHost',
                'Oracle', 'Tempest', 'RoachBurrowed', 'WidowMine', 'Viper', 'Lurker', 'OverlordTransport', 'Queen',
                'InfestorBurrowed'
            ],
            'target_type': [],
            'target_type_name': []
        },
        17: {
            'action_name': 'HoldPosition_quick',
            'selected_type': [
                128, 129, 4, 134, 7, 688, 9, 10, 139, 268, 13, 12, 15, 14, 17, 16, 140, 141, 144, 150, 489, 801, 33, 35,
                34, 36, 289, 43, 44, 45, 46, 48, 49, 50, 51, 52, 53, 54, 311, 55, 57, 56, 689, 691, 692, 693, 694, 73,
                74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 733, 484, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113,
                114, 494, 495, 496, 118, 498, 499, 502, 892, 893, 126, 127
            ],
            'selected_type_name': [
                'OverseerCocoon', 'Overseer', 'Colossus', 'OrbitalCommandFlying', 'InfestedTerran', 'Ravager',
                'Baneling', 'Mothership', 'SpineCrawlerUprooted', 'MULE', 'ChangelingZealot', 'Changeling',
                'ChangelingMarine', 'ChangelingMarineShield', 'ChangelingZergling', 'ChangelingZerglingWings',
                'SporeCrawlerUprooted', 'Archon', 'GhostAlternate', 'InfestedTerranCocoon', 'Locust', 'AdeptPhaseShift',
                'SiegeTank', 'VikingFighter', 'VikingAssault', 'CommandCenterFlying', 'Broodling', 'FactoryFlying',
                'StarportFlying', 'SCV', 'BarracksFlying', 'Marine', 'Reaper', 'Ghost', 'Marauder', 'Thor', 'Hellion',
                'Medivac', 'Adept', 'Banshee', 'Battlecruiser', 'Raven', 'Liberator', 'ThorHighImpactMode', 'Cyclone',
                'LocustFlying', 'Disruptor', 'Zealot', 'Stalker', 'HighTemplar', 'DarkTemplar', 'Sentry', 'Phoenix',
                'Carrier', 'VoidRay', 'WarpPrism', 'Observer', 'Immortal', 'Probe', 'DisruptorPhased', 'Hellbat',
                'Drone', 'Zergling', 'Overlord', 'Hydralisk', 'Mutalisk', 'Ultralisk', 'Roach', 'Infestor', 'Corruptor',
                'BroodLordCocoon', 'BroodLord', 'SwarmHost', 'Oracle', 'Tempest', 'RoachBurrowed', 'WidowMine', 'Viper',
                'Lurker', 'OverlordTransportCocoon', 'OverlordTransport', 'Queen', 'InfestorBurrowed'
            ],
            'target_type': [],
            'target_type_name': []
        },
        19: {
            'action_name': 'Research_PhoenixAnionPulseCrystals_quick',
            'selected_type': [64],
            'selected_type_name': ['FleetBeacon'],
            'target_type': [],
            'target_type_name': []
        },
        20: {
            'action_name': 'Effect_GuardianShield_quick',
            'selected_type': [77],
            'selected_type_name': ['Sentry'],
            'target_type': [],
            'target_type_name': []
        },
        21: {
            'action_name': 'Train_Mothership_quick',
            'selected_type': [59],
            'selected_type_name': ['Nexus'],
            'target_type': [],
            'target_type_name': []
        },
        22: {
            'action_name': 'Hallucination_Archon_quick',
            'selected_type': [77],
            'selected_type_name': ['Sentry'],
            'target_type': [],
            'target_type_name': []
        },
        23: {
            'action_name': 'Hallucination_Colossus_quick',
            'selected_type': [77],
            'selected_type_name': ['Sentry'],
            'target_type': [],
            'target_type_name': []
        },
        24: {
            'action_name': 'Hallucination_HighTemplar_quick',
            'selected_type': [77],
            'selected_type_name': ['Sentry'],
            'target_type': [],
            'target_type_name': []
        },
        25: {
            'action_name': 'Hallucination_Immortal_quick',
            'selected_type': [77],
            'selected_type_name': ['Sentry'],
            'target_type': [],
            'target_type_name': []
        },
        26: {
            'action_name': 'Hallucination_Phoenix_quick',
            'selected_type': [77],
            'selected_type_name': ['Sentry'],
            'target_type': [],
            'target_type_name': []
        },
        27: {
            'action_name': 'Hallucination_Probe_quick',
            'selected_type': [77],
            'selected_type_name': ['Sentry'],
            'target_type': [],
            'target_type_name': []
        },
        28: {
            'action_name': 'Hallucination_Stalker_quick',
            'selected_type': [77],
            'selected_type_name': ['Sentry'],
            'target_type': [],
            'target_type_name': []
        },
        29: {
            'action_name': 'Hallucination_VoidRay_quick',
            'selected_type': [77],
            'selected_type_name': ['Sentry'],
            'target_type': [],
            'target_type_name': []
        },
        30: {
            'action_name': 'Hallucination_WarpPrism_quick',
            'selected_type': [77],
            'selected_type_name': ['Sentry'],
            'target_type': [],
            'target_type_name': []
        },
        31: {
            'action_name': 'Hallucination_Zealot_quick',
            'selected_type': [77],
            'selected_type_name': ['Sentry'],
            'target_type': [],
            'target_type_name': []
        },
        32: {
            'action_name': 'Effect_GravitonBeam_unit',
            'selected_type': [73, 78],
            'selected_type_name': ['Zealot', 'Phoenix'],
            'target_type': [
                7, 8, 9, 13, 151, 289, 687, 688, 311, 73, 74, 75, 76, 77, 83, 84, 119, 103, 104, 105, 489, 107, 493,
                110, 494, 111, 116, 501, 502, 503, 117, 118, 125, 126, 127
            ],
            'target_type_name': [
                'InfestedTerran', 'BanelingCocoon', 'Baneling', 'ChangelingZealot', 'Larva', 'Broodling',
                'RavagerCocoon', 'Ravager', 'Adept', 'Zealot', 'Stalker', 'HighTemplar', 'DarkTemplar', 'Sentry',
                'Immortal', 'Probe', 'ZerglingBurrowed', 'Cocoon', 'Drone', 'Zergling', 'Locust', 'Hydralisk',
                'SwarmHostBurrowed', 'Roach', 'SwarmHost', 'Infestor', 'DroneBurrowed', 'LurkerCocoon', 'Lurker',
                'LurkerBurrowed', 'HydraliskBurrowed', 'RoachBurrowed', 'QueenBurrowed', 'Queen', 'InfestorBurrowed'
            ]
        },
        34: {
            'action_name': 'Build_Nexus_pt',
            'selected_type': [84],
            'selected_type_name': ['Probe'],
            'target_type': [],
            'target_type_name': []
        },
        35: {
            'action_name': 'Build_Pylon_pt',
            'selected_type': [84],
            'selected_type_name': ['Probe'],
            'target_type': [],
            'target_type_name': []
        },
        36: {
            'action_name': 'Build_Assimilator_unit',
            'selected_type': [84],
            'selected_type_name': ['Probe'],
            'target_type': [880, 881, 342, 343, 665],
            'target_type_name': [
                'PurifierVespeneGeyser', 'ShakurasVespeneGeyser', 'VespeneGeyser', 'SpacePlatformGeyser',
                'LabMineralField'
            ]
        },
        37: {
            'action_name': 'Build_Gateway_pt',
            'selected_type': [84],
            'selected_type_name': ['Probe'],
            'target_type': [],
            'target_type_name': []
        },
        38: {
            'action_name': 'Build_Forge_pt',
            'selected_type': [84],
            'selected_type_name': ['Probe'],
            'target_type': [],
            'target_type_name': []
        },
        39: {
            'action_name': 'Build_FleetBeacon_pt',
            'selected_type': [84],
            'selected_type_name': ['Probe'],
            'target_type': [],
            'target_type_name': []
        },
        40: {
            'action_name': 'Build_TwilightCouncil_pt',
            'selected_type': [84],
            'selected_type_name': ['Probe'],
            'target_type': [],
            'target_type_name': []
        },
        41: {
            'action_name': 'Build_PhotonCannon_pt',
            'selected_type': [84],
            'selected_type_name': ['Probe'],
            'target_type': [],
            'target_type_name': []
        },
        42: {
            'action_name': 'Build_Stargate_pt',
            'selected_type': [84],
            'selected_type_name': ['Probe'],
            'target_type': [],
            'target_type_name': []
        },
        43: {
            'action_name': 'Build_TemplarArchive_pt',
            'selected_type': [84],
            'selected_type_name': ['Probe'],
            'target_type': [],
            'target_type_name': []
        },
        44: {
            'action_name': 'Build_DarkShrine_pt',
            'selected_type': [84],
            'selected_type_name': ['Probe'],
            'target_type': [],
            'target_type_name': []
        },
        45: {
            'action_name': 'Build_RoboticsBay_pt',
            'selected_type': [84],
            'selected_type_name': ['Probe'],
            'target_type': [],
            'target_type_name': []
        },
        46: {
            'action_name': 'Build_RoboticsFacility_pt',
            'selected_type': [84],
            'selected_type_name': ['Probe'],
            'target_type': [],
            'target_type_name': []
        },
        47: {
            'action_name': 'Build_CyberneticsCore_pt',
            'selected_type': [84],
            'selected_type_name': ['Probe'],
            'target_type': [],
            'target_type_name': []
        },
        48: {
            'action_name': 'Build_ShieldBattery_pt',
            'selected_type': [84],
            'selected_type_name': ['Probe'],
            'target_type': [],
            'target_type_name': []
        },
        49: {
            'action_name': 'Train_Zealot_quick',
            'selected_type': [62],
            'selected_type_name': ['Gateway'],
            'target_type': [],
            'target_type_name': []
        },
        50: {
            'action_name': 'Train_Stalker_quick',
            'selected_type': [62],
            'selected_type_name': ['Gateway'],
            'target_type': [],
            'target_type_name': []
        },
        51: {
            'action_name': 'Train_HighTemplar_quick',
            'selected_type': [62],
            'selected_type_name': ['Gateway'],
            'target_type': [],
            'target_type_name': []
        },
        52: {
            'action_name': 'Train_DarkTemplar_quick',
            'selected_type': [62],
            'selected_type_name': ['Gateway'],
            'target_type': [],
            'target_type_name': []
        },
        53: {
            'action_name': 'Train_Sentry_quick',
            'selected_type': [62],
            'selected_type_name': ['Gateway'],
            'target_type': [],
            'target_type_name': []
        },
        54: {
            'action_name': 'Train_Adept_quick',
            'selected_type': [62],
            'selected_type_name': ['Gateway'],
            'target_type': [],
            'target_type_name': []
        },
        55: {
            'action_name': 'Train_Phoenix_quick',
            'selected_type': [67],
            'selected_type_name': ['Stargate'],
            'target_type': [],
            'target_type_name': []
        },
        56: {
            'action_name': 'Train_Carrier_quick',
            'selected_type': [67],
            'selected_type_name': ['Stargate'],
            'target_type': [],
            'target_type_name': []
        },
        57: {
            'action_name': 'Train_VoidRay_quick',
            'selected_type': [67],
            'selected_type_name': ['Stargate'],
            'target_type': [],
            'target_type_name': []
        },
        58: {
            'action_name': 'Train_Oracle_quick',
            'selected_type': [67],
            'selected_type_name': ['Stargate'],
            'target_type': [],
            'target_type_name': []
        },
        59: {
            'action_name': 'Train_Tempest_quick',
            'selected_type': [67],
            'selected_type_name': ['Stargate'],
            'target_type': [],
            'target_type_name': []
        },
        60: {
            'action_name': 'Train_WarpPrism_quick',
            'selected_type': [71],
            'selected_type_name': ['RoboticsFacility'],
            'target_type': [],
            'target_type_name': []
        },
        61: {
            'action_name': 'Train_Observer_quick',
            'selected_type': [71],
            'selected_type_name': ['RoboticsFacility'],
            'target_type': [],
            'target_type_name': []
        },
        62: {
            'action_name': 'Train_Colossus_quick',
            'selected_type': [71],
            'selected_type_name': ['RoboticsFacility'],
            'target_type': [],
            'target_type_name': []
        },
        63: {
            'action_name': 'Train_Immortal_quick',
            'selected_type': [71],
            'selected_type_name': ['RoboticsFacility'],
            'target_type': [],
            'target_type_name': []
        },
        64: {
            'action_name': 'Train_Probe_quick',
            'selected_type': [59],
            'selected_type_name': ['Nexus'],
            'target_type': [],
            'target_type_name': []
        },
        65: {
            'action_name': 'Effect_PsiStorm_pt',
            'selected_type': [75],
            'selected_type_name': ['HighTemplar'],
            'target_type': [],
            'target_type_name': []
        },
        66: {
            'action_name': 'Build_Interceptors_quick',
            'selected_type': [79],
            'selected_type_name': ['Carrier'],
            'target_type': [],
            'target_type_name': []
        },
        67: {
            'action_name': 'Research_GraviticBooster_quick',
            'selected_type': [70],
            'selected_type_name': ['RoboticsBay'],
            'target_type': [],
            'target_type_name': []
        },
        68: {
            'action_name': 'Research_GraviticDrive_quick',
            'selected_type': [70],
            'selected_type_name': ['RoboticsBay'],
            'target_type': [],
            'target_type_name': []
        },
        69: {
            'action_name': 'Research_ExtendedThermalLance_quick',
            'selected_type': [70],
            'selected_type_name': ['RoboticsBay'],
            'target_type': [],
            'target_type_name': []
        },
        70: {
            'action_name': 'Research_PsiStorm_quick',
            'selected_type': [68],
            'selected_type_name': ['TemplarArchive'],
            'target_type': [],
            'target_type_name': []
        },
        71: {
            'action_name': 'TrainWarp_Zealot_pt',
            'selected_type': [133],
            'selected_type_name': ['WarpGate'],
            'target_type': [],
            'target_type_name': []
        },
        72: {
            'action_name': 'TrainWarp_Stalker_pt',
            'selected_type': [133],
            'selected_type_name': ['WarpGate'],
            'target_type': [],
            'target_type_name': []
        },
        73: {
            'action_name': 'TrainWarp_HighTemplar_pt',
            'selected_type': [133],
            'selected_type_name': ['WarpGate'],
            'target_type': [],
            'target_type_name': []
        },
        74: {
            'action_name': 'TrainWarp_DarkTemplar_pt',
            'selected_type': [133],
            'selected_type_name': ['WarpGate'],
            'target_type': [],
            'target_type_name': []
        },
        75: {
            'action_name': 'TrainWarp_Sentry_pt',
            'selected_type': [133],
            'selected_type_name': ['WarpGate'],
            'target_type': [],
            'target_type_name': []
        },
        76: {
            'action_name': 'TrainWarp_Adept_pt',
            'selected_type': [133],
            'selected_type_name': ['WarpGate'],
            'target_type': [],
            'target_type_name': []
        },
        77: {
            'action_name': 'Morph_WarpGate_quick',
            'selected_type': [62],
            'selected_type_name': ['Gateway'],
            'target_type': [],
            'target_type_name': []
        },
        78: {
            'action_name': 'Morph_Gateway_quick',
            'selected_type': [133],
            'selected_type_name': ['WarpGate'],
            'target_type': [],
            'target_type_name': []
        },
        79: {
            'action_name': 'Effect_ForceField_pt',
            'selected_type': [77],
            'selected_type_name': ['Sentry'],
            'target_type': [],
            'target_type_name': []
        },
        80: {
            'action_name': 'Morph_WarpPrismPhasingMode_quick',
            'selected_type': [81],
            'selected_type_name': ['WarpPrism'],
            'target_type': [],
            'target_type_name': []
        },
        81: {
            'action_name': 'Morph_WarpPrismTransportMode_quick',
            'selected_type': [136],
            'selected_type_name': ['WarpPrismPhasing'],
            'target_type': [],
            'target_type_name': []
        },
        82: {
            'action_name': 'Research_WarpGate_quick',
            'selected_type': [72],
            'selected_type_name': ['CyberneticsCore'],
            'target_type': [],
            'target_type_name': []
        },
        83: {
            'action_name': 'Research_Charge_quick',
            'selected_type': [65],
            'selected_type_name': ['TwilightCouncil'],
            'target_type': [],
            'target_type_name': []
        },
        84: {
            'action_name': 'Research_Blink_quick',
            'selected_type': [65],
            'selected_type_name': ['TwilightCouncil'],
            'target_type': [],
            'target_type_name': []
        },
        85: {
            'action_name': 'Research_AdeptResonatingGlaives_quick',
            'selected_type': [65],
            'selected_type_name': ['TwilightCouncil'],
            'target_type': [],
            'target_type_name': []
        },
        86: {
            'action_name': 'Morph_Archon_quick',
            'selected_type': [75, 76],
            'selected_type_name': ['HighTemplar', 'DarkTemplar'],
            'target_type': [],
            'target_type_name': []
        },
        87: {
            'action_name': 'Behavior_BuildingAttackOn_quick',
            'selected_type': [9],
            'selected_type_name': ['Baneling'],
            'target_type': [],
            'target_type_name': []
        },
        88: {
            'action_name': 'Behavior_BuildingAttackOff_quick',
            'selected_type': [9],
            'selected_type_name': ['Baneling'],
            'target_type': [],
            'target_type_name': []
        },
        89: {
            'action_name': 'Hallucination_Oracle_quick',
            'selected_type': [77],
            'selected_type_name': ['Sentry'],
            'target_type': [],
            'target_type_name': []
        },
        90: {
            'action_name': 'Effect_OracleRevelation_pt',
            'selected_type': [495],
            'selected_type_name': ['Oracle'],
            'target_type': [],
            'target_type_name': []
        },
        92: {
            'action_name': 'Hallucination_Disruptor_quick',
            'selected_type': [77],
            'selected_type_name': ['Sentry'],
            'target_type': [],
            'target_type_name': []
        },
        93: {
            'action_name': 'Hallucination_Adept_quick',
            'selected_type': [77],
            'selected_type_name': ['Sentry'],
            'target_type': [],
            'target_type_name': []
        },
        94: {
            'action_name': 'Effect_VoidRayPrismaticAlignment_quick',
            'selected_type': [80],
            'selected_type_name': ['VoidRay'],
            'target_type': [],
            'target_type_name': []
        },
        95: {
            'action_name': 'Build_StasisTrap_pt',
            'selected_type': [495],
            'selected_type_name': ['Oracle'],
            'target_type': [],
            'target_type_name': []
        },
        96: {
            'action_name': 'Effect_AdeptPhaseShift_pt',
            'selected_type': [311],
            'selected_type_name': ['Adept'],
            'target_type': [],
            'target_type_name': []
        },
        97: {
            'action_name': 'Research_ShadowStrike_quick',
            'selected_type': [69],
            'selected_type_name': ['DarkShrine'],
            'target_type': [],
            'target_type_name': []
        },
        98: {
            'action_name': 'Cancel_quick',
            'selected_type': [
                128, 137, 138, 139, 140, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 801, 37, 38, 39, 40, 41,
                42, 687, 50, 692, 311, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 78, 80, 86, 87, 88, 89,
                90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 111, 495, 113, 501, 1910, 504, 892, 127
            ],
            'selected_type_name': [
                'OverseerCocoon', 'CreepTumorBurrowed', 'CreepTumorQueen', 'SpineCrawlerUprooted',
                'SporeCrawlerUprooted', 'CommandCenter', 'SupplyDepot', 'Refinery', 'Barracks', 'EngineeringBay',
                'MissileTurret', 'Bunker', 'SensorTower', 'GhostAcademy', 'Factory', 'Starport', 'Armory', 'FusionCore',
                'AdeptPhaseShift', 'BarracksTechLab', 'BarracksReactor', 'FactoryTechLab', 'FactoryReactor',
                'StarportTechLab', 'StarportReactor', 'RavagerCocoon', 'Ghost', 'Cyclone', 'Adept', 'Nexus', 'Pylon',
                'Assimilator', 'Gateway', 'Forge', 'FleetBeacon', 'TwilightCouncil', 'PhotonCannon', 'Stargate',
                'TemplarArchive', 'DarkShrine', 'RoboticsBay', 'RoboticsFacility', 'CyberneticsCore', 'Phoenix',
                'VoidRay', 'Hatchery', 'CreepTumor', 'Extractor', 'SpawningPool', 'EvolutionChamber', 'HydraliskDen',
                'Spire', 'UltraliskCavern', 'InfestationPit', 'NydusNetwork', 'BanelingNest', 'RoachWarren',
                'SpineCrawler', 'SporeCrawler', 'Lair', 'Infestor', 'Oracle', 'BroodLordCocoon', 'LurkerCocoon',
                'ShieldBattery', 'LurkerDen', 'OverlordTransportCocoon', 'InfestorBurrowed'
            ],
            'target_type': [],
            'target_type_name': []
        },
        99: {
            'action_name': 'Halt_quick',
            'selected_type': [45, 18, 19, 20, 21, 22, 23, 24, 27, 28, 29, 30],
            'selected_type_name': [
                'SCV', 'CommandCenter', 'SupplyDepot', 'Refinery', 'Barracks', 'EngineeringBay', 'MissileTurret',
                'Bunker', 'Factory', 'Starport', 'Armory', 'FusionCore'
            ],
            'target_type': [],
            'target_type_name': []
        },
        100: {
            'action_name': 'UnloadAll_quick',
            'selected_type': [130, 36, 142, 18, 24, 95],
            'selected_type_name': [
                'PlanetaryFortress', 'CommandCenterFlying', 'NydusCanal', 'CommandCenter', 'Bunker', 'NydusNetwork'
            ],
            'target_type': [],
            'target_type_name': []
        },
        101: {
            'action_name': 'Stop_quick',
            'selected_type': [
                4, 7, 9, 10, 12, 13, 14, 15, 16, 17, 23, 24, 32, 33, 34, 35, 36, 43, 44, 45, 46, 48, 49, 50, 51, 52, 53,
                54, 55, 56, 57, 66, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 95, 98, 99, 104, 105, 106, 107, 108,
                109, 110, 111, 112, 114, 118, 125, 126, 127, 129, 130, 134, 136, 139, 140, 141, 142, 144, 688, 689, 691,
                692, 693, 694, 733, 734, 268, 801, 289, 311, 1910, 1911, 1912, 893, 484, 489, 494, 495, 496, 498, 499,
                502, 503
            ],
            'selected_type_name': [
                'Colossus', 'InfestedTerran', 'Baneling', 'Mothership', 'Changeling', 'ChangelingZealot',
                'ChangelingMarineShield', 'ChangelingMarine', 'ChangelingZerglingWings', 'ChangelingZergling',
                'MissileTurret', 'Bunker', 'SiegeTankSieged', 'SiegeTank', 'VikingAssault', 'VikingFighter',
                'CommandCenterFlying', 'FactoryFlying', 'StarportFlying', 'SCV', 'BarracksFlying', 'Marine', 'Reaper',
                'Ghost', 'Marauder', 'Thor', 'Hellion', 'Medivac', 'Banshee', 'Raven', 'Battlecruiser', 'PhotonCannon',
                'Zealot', 'Stalker', 'HighTemplar', 'DarkTemplar', 'Sentry', 'Phoenix', 'Carrier', 'VoidRay',
                'WarpPrism', 'Observer', 'Immortal', 'Probe', 'NydusNetwork', 'SpineCrawler', 'SporeCrawler', 'Drone',
                'Zergling', 'Overlord', 'Hydralisk', 'Mutalisk', 'Ultralisk', 'Roach', 'Infestor', 'Corruptor',
                'BroodLord', 'RoachBurrowed', 'QueenBurrowed', 'Queen', 'InfestorBurrowed', 'Overseer',
                'PlanetaryFortress', 'OrbitalCommandFlying', 'WarpPrismPhasing', 'SpineCrawlerUprooted',
                'SporeCrawlerUprooted', 'Archon', 'NydusCanal', 'GhostAlternate', 'Ravager', 'Liberator',
                'ThorHighImpactMode', 'Cyclone', 'LocustFlying', 'Disruptor', 'DisruptorPhased', 'LiberatorAG', 'MULE',
                'AdeptPhaseShift', 'Broodling', 'Adept', 'ShieldBattery', 'ObserverSurveillanceMode',
                'OverseerOversightMode', 'OverlordTransport', 'Hellbat', 'Locust', 'SwarmHost', 'Oracle', 'Tempest',
                'WidowMine', 'Viper', 'Lurker', 'LurkerBurrowed'
            ],
            'target_type': [],
            'target_type_name': []
        },
        102: {
            'action_name': 'Harvest_Gather_unit',
            'selected_type': [104, 45, 84, 268],
            'selected_type_name': ['Drone', 'SCV', 'Probe', 'MULE'],
            'target_type': [483, 61, 884, 885, 341, 20, 88, 665, 666, 796, 797],
            'target_type_name': [
                'MineralField750', 'Assimilator', 'PurifierMineralField', 'PurifierMineralField750', 'MineralField',
                'Refinery', 'Extractor', 'LabMineralField', 'LabMineralField750', 'PurifierRichMineralField',
                'PurifierRichMineralField750'
            ]
        },
        103: {
            'action_name': 'Harvest_Return_quick',
            'selected_type': [104, 45, 84, 268],
            'selected_type_name': ['Drone', 'SCV', 'Probe', 'MULE'],
            'target_type': [],
            'target_type_name': []
        },
        104: {
            'action_name': 'Load_unit',
            'selected_type': [136, 142, 81, 54, 24, 893, 95],
            'selected_type_name': [
                'WarpPrismPhasing', 'NydusCanal', 'WarpPrism', 'Medivac', 'Bunker', 'OverlordTransport', 'NydusNetwork'
            ],
            'target_type': [
                4, 9, 141, 33, 34, 45, 48, 49, 688, 51, 52, 53, 73, 74, 75, 76, 77, 83, 84, 484, 104, 105, 107, 110,
                498, 502, 126
            ],
            'target_type_name': [
                'Colossus', 'Baneling', 'Archon', 'SiegeTank', 'VikingAssault', 'SCV', 'Marine', 'Reaper', 'Ravager',
                'Marauder', 'Thor', 'Hellion', 'Zealot', 'Stalker', 'HighTemplar', 'DarkTemplar', 'Sentry', 'Immortal',
                'Probe', 'Hellbat', 'Drone', 'Zergling', 'Hydralisk', 'Roach', 'WidowMine', 'Lurker', 'Queen'
            ]
        },
        105: {
            'action_name': 'UnloadAllAt_pt',
            'selected_type': [136, 81, 893, 54],
            'selected_type_name': ['WarpPrismPhasing', 'WarpPrism', 'OverlordTransport', 'Medivac'],
            'target_type': [],
            'target_type_name': []
        },
        106: {
            'action_name': 'Rally_Units_pt',
            'selected_type': [67, 100, 101, 71, 103, 73, 74, 141, 142, 21, 86, 24, 27, 28, 62, 95],
            'selected_type_name': [
                'Stargate', 'Lair', 'Hive', 'RoboticsFacility', 'Cocoon', 'Zealot', 'Stalker', 'Archon', 'NydusCanal',
                'Barracks', 'Hatchery', 'Bunker', 'Factory', 'Starport', 'Gateway', 'NydusNetwork'
            ],
            'target_type': [],
            'target_type_name': []
        },
        107: {
            'action_name': 'Rally_Units_unit',
            'selected_type': [67, 100, 101, 71, 103, 73, 74, 59, 141, 142, 21, 86, 24, 27, 28, 62],
            'selected_type_name': [
                'Stargate', 'Lair', 'Hive', 'RoboticsFacility', 'Cocoon', 'Zealot', 'Stalker', 'Nexus', 'Archon',
                'NydusCanal', 'Barracks', 'Hatchery', 'Bunker', 'Factory', 'Starport', 'Gateway'
            ],
            'target_type': [
                4, 8, 9, 10, 18, 19, 21, 24, 27, 28, 29, 32, 33, 35, 38, 39, 43, 45, 47, 48, 49, 50, 51, 52, 53, 54, 55,
                57, 59, 60, 61, 62, 63, 66, 67, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 83, 84, 86, 87, 88, 89, 90,
                91, 92, 94, 95, 96, 97, 98, 99, 100, 101, 103, 104, 105, 106, 107, 108, 109, 110, 112, 114, 630, 118,
                126, 127, 639, 129, 641, 128, 132, 133, 137, 138, 139, 140, 141, 142, 146, 147, 149, 151, 665, 666, 688,
                689, 692, 734, 796, 311, 341, 342, 343, 880, 881, 884, 885, 1910, 472, 474, 483, 484, 485, 493, 494,
                495, 496, 498, 500, 501, 502, 503
            ],
            'target_type_name': [
                'Colossus', 'BanelingCocoon', 'Baneling', 'Mothership', 'CommandCenter', 'SupplyDepot', 'Barracks',
                'Bunker', 'Factory', 'Starport', 'Armory', 'SiegeTankSieged', 'SiegeTank', 'VikingFighter',
                'BarracksReactor', 'FactoryTechLab', 'FactoryFlying', 'SCV', 'SupplyDepotLowered', 'Marine', 'Reaper',
                'Ghost', 'Marauder', 'Thor', 'Hellion', 'Medivac', 'Banshee', 'Battlecruiser', 'Nexus', 'Pylon',
                'Assimilator', 'Gateway', 'Forge', 'PhotonCannon', 'Stargate', 'RoboticsFacility', 'CyberneticsCore',
                'Zealot', 'Stalker', 'HighTemplar', 'DarkTemplar', 'Sentry', 'Phoenix', 'Carrier', 'VoidRay',
                'WarpPrism', 'Immortal', 'Probe', 'Hatchery', 'CreepTumor', 'Extractor', 'SpawningPool',
                'EvolutionChamber', 'HydraliskDen', 'Spire', 'InfestationPit', 'NydusNetwork', 'BanelingNest',
                'RoachWarren', 'SpineCrawler', 'SporeCrawler', 'Lair', 'Hive', 'Cocoon', 'Drone', 'Zergling',
                'Overlord', 'Hydralisk', 'Mutalisk', 'Ultralisk', 'Roach', 'Corruptor', 'BroodLord',
                'DestructibleCityDebrisHugeDiagonalBLUR', 'RoachBurrowed', 'Queen', 'InfestorBurrowed',
                'DestructibleRockEx16x6', 'Overseer', 'DestructibleRockEx1DiagonalHugeBLUR', 'OverseerCocoon',
                'OrbitalCommand', 'WarpGate', 'CreepTumorBurrowed', 'CreepTumorQueen', 'SpineCrawlerUprooted',
                'SporeCrawlerUprooted', 'Archon', 'NydusCanal', 'RichMineralField', 'RichMineralField750',
                'XelNagaTower', 'Larva', 'LabMineralField', 'LabMineralField750', 'Ravager', 'Liberator', 'Cyclone',
                'LiberatorAG', 'PurifierRichMineralField', 'Adept', 'MineralField', 'VespeneGeyser',
                'SpacePlatformGeyser', 'PurifierVespeneGeyser', 'ShakurasVespeneGeyser', 'PurifierMineralField',
                'PurifierMineralField750', 'ShieldBattery', 'UnbuildableRocksDestructible',
                'UnbuildablePlatesDestructible', 'MineralField750', 'Hellbat', 'CollapsibleTerranTowerDebris',
                'SwarmHostBurrowed', 'SwarmHost', 'Oracle', 'Tempest', 'WidowMine', 'WidowMineBurrowed', 'LurkerCocoon',
                'Lurker', 'LurkerBurrowed'
            ]
        },
        109: {
            'action_name': 'Effect_Repair_unit',
            'selected_type': [268, 45],
            'selected_type_name': ['MULE', 'SCV'],
            'target_type': [
                130, 132, 5, 134, 6, 268, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38,
                39, 40, 41, 42, 43, 44, 45, 46, 47, 689, 691, 692, 53, 54, 55, 56, 57, 52, 734, 484, 498, 500
            ],
            'target_type_name': [
                'PlanetaryFortress', 'OrbitalCommand', 'TechLab', 'OrbitalCommandFlying', 'Reactor', 'MULE',
                'CommandCenter', 'SupplyDepot', 'Refinery', 'Barracks', 'EngineeringBay', 'MissileTurret', 'Bunker',
                'GhostAcademy', 'Factory', 'Starport', 'Armory', 'FusionCore', 'SiegeTankSieged', 'SiegeTank',
                'VikingAssault', 'VikingFighter', 'CommandCenterFlying', 'BarracksTechLab', 'BarracksReactor',
                'FactoryTechLab', 'FactoryReactor', 'StarportTechLab', 'StarportReactor', 'FactoryFlying',
                'StarportFlying', 'SCV', 'BarracksFlying', 'SupplyDepotLowered', 'Liberator', 'ThorHighImpactMode',
                'Cyclone', 'Hellion', 'Medivac', 'Banshee', 'Raven', 'Battlecruiser', 'Thor', 'LiberatorAG', 'Hellbat',
                'WidowMine', 'WidowMineBurrowed'
            ]
        },
        110: {
            'action_name': 'Effect_MassRecall_pt',
            'selected_type': [10, 59],
            'selected_type_name': ['Mothership', 'Nexus'],
            'target_type': [],
            'target_type_name': []
        },
        111: {
            'action_name': 'Effect_Blink_pt',
            'selected_type': [74, 76],
            'selected_type_name': ['Stalker', 'DarkTemplar'],
            'target_type': [],
            'target_type_name': []
        },
        114: {
            'action_name': 'Rally_Workers_pt',
            'selected_type': [130, 100, 132, 101, 71, 18, 21, 86, 59, 28, 62],
            'selected_type_name': [
                'PlanetaryFortress', 'Lair', 'OrbitalCommand', 'Hive', 'RoboticsFacility', 'CommandCenter', 'Barracks',
                'Hatchery', 'Nexus', 'Starport', 'Gateway'
            ],
            'target_type': [],
            'target_type_name': []
        },
        115: {
            'action_name': 'Rally_Workers_unit',
            'selected_type': [130, 100, 132, 101, 18, 86, 59, 28],
            'selected_type_name': [
                'PlanetaryFortress', 'Lair', 'OrbitalCommand', 'Hive', 'CommandCenter', 'Hatchery', 'Nexus', 'Starport'
            ],
            'target_type': [
                132, 134, 268, 146, 19, 20, 18, 21, 147, 151, 665, 666, 796, 797, 45, 47, 57, 59, 60, 61, 63, 72, 74,
                78, 79, 80, 84, 341, 86, 343, 342, 88, 90, 475, 91, 92, 94, 95, 96, 97, 89, 483, 100, 99, 104, 105, 107,
                108, 880, 881, 884, 885, 893, 126
            ],
            'target_type_name': [
                'OrbitalCommand', 'OrbitalCommandFlying', 'MULE', 'RichMineralField', 'SupplyDepot', 'Refinery',
                'CommandCenter', 'Barracks', 'RichMineralField750', 'Larva', 'LabMineralField', 'LabMineralField750',
                'PurifierRichMineralField', 'PurifierRichMineralField750', 'SCV', 'SupplyDepotLowered', 'Battlecruiser',
                'Nexus', 'Pylon', 'Assimilator', 'Forge', 'CyberneticsCore', 'Stalker', 'Phoenix', 'Carrier', 'VoidRay',
                'Probe', 'MineralField', 'Hatchery', 'SpacePlatformGeyser', 'VespeneGeyser', 'Extractor',
                'EvolutionChamber', 'Debris2x2NonConjoined', 'HydraliskDen', 'Spire', 'InfestationPit', 'NydusNetwork',
                'BanelingNest', 'RoachWarren', 'SpawningPool', 'MineralField750', 'Lair', 'SporeCrawler', 'Drone',
                'Zergling', 'Hydralisk', 'Mutalisk', 'PurifierVespeneGeyser', 'ShakurasVespeneGeyser',
                'PurifierMineralField', 'PurifierMineralField750', 'OverlordTransport', 'Queen'
            ]
        },
        116: {
            'action_name': 'Research_ProtossAirArmor_quick',
            'selected_type': [72],
            'selected_type_name': ['CyberneticsCore'],
            'target_type': [],
            'target_type_name': []
        },
        117: {
            'action_name': 'Research_ProtossAirWeapons_quick',
            'selected_type': [72],
            'selected_type_name': ['CyberneticsCore'],
            'target_type': [],
            'target_type_name': []
        },
        118: {
            'action_name': 'Research_ProtossGroundArmor_quick',
            'selected_type': [63],
            'selected_type_name': ['Forge'],
            'target_type': [],
            'target_type_name': []
        },
        119: {
            'action_name': 'Research_ProtossGroundWeapons_quick',
            'selected_type': [63],
            'selected_type_name': ['Forge'],
            'target_type': [],
            'target_type_name': []
        },
        120: {
            'action_name': 'Research_ProtossShields_quick',
            'selected_type': [63],
            'selected_type_name': ['Forge'],
            'target_type': [],
            'target_type_name': []
        },
        121: {
            'action_name': 'Morph_ObserverMode_quick',
            'selected_type': [1911],
            'selected_type_name': ['ObserverSurveillanceMode'],
            'target_type': [],
            'target_type_name': []
        },
        122: {
            'action_name': 'Effect_ChronoBoostEnergyCost_unit',
            'selected_type': [59],
            'selected_type_name': ['Nexus'],
            'target_type': [64, 65, 67, 68, 133, 70, 71, 72, 69, 59, 62, 63],
            'target_type_name': [
                'FleetBeacon', 'TwilightCouncil', 'Stargate', 'TemplarArchive', 'WarpGate', 'RoboticsBay',
                'RoboticsFacility', 'CyberneticsCore', 'DarkShrine', 'Nexus', 'Gateway', 'Forge'
            ]
        },
        129: {
            'action_name': 'Cancel_Last_quick',
            'selected_type': [
                130, 132, 8, 18, 21, 22, 26, 27, 28, 29, 30, 37, 39, 41, 59, 62, 63, 64, 65, 67, 68, 69, 70, 71, 72, 79,
                86, 89, 90, 91, 92, 93, 94, 96, 97, 100, 101, 102, 103
            ],
            'selected_type_name': [
                'PlanetaryFortress', 'OrbitalCommand', 'BanelingCocoon', 'CommandCenter', 'Barracks', 'EngineeringBay',
                'GhostAcademy', 'Factory', 'Starport', 'Armory', 'FusionCore', 'BarracksTechLab', 'FactoryTechLab',
                'StarportTechLab', 'Nexus', 'Gateway', 'Forge', 'FleetBeacon', 'TwilightCouncil', 'Stargate',
                'TemplarArchive', 'DarkShrine', 'RoboticsBay', 'RoboticsFacility', 'CyberneticsCore', 'Carrier',
                'Hatchery', 'SpawningPool', 'EvolutionChamber', 'HydraliskDen', 'Spire', 'UltraliskCavern',
                'InfestationPit', 'BanelingNest', 'RoachWarren', 'Lair', 'Hive', 'GreaterSpire', 'Cocoon'
            ],
            'target_type': [],
            'target_type_name': []
        },
        157: {
            'action_name': 'Effect_Feedback_unit',
            'selected_type': [75],
            'selected_type_name': ['HighTemplar'],
            'target_type': [129, 75, 78, 111, 499, 1912, 126, 127],
            'target_type_name': [
                'Overseer', 'HighTemplar', 'Phoenix', 'Infestor', 'Viper', 'OverseerOversightMode', 'Queen',
                'InfestorBurrowed'
            ]
        },
        158: {
            'action_name': 'Behavior_PulsarBeamOff_quick',
            'selected_type': [495],
            'selected_type_name': ['Oracle'],
            'target_type': [],
            'target_type_name': []
        },
        159: {
            'action_name': 'Behavior_PulsarBeamOn_quick',
            'selected_type': [495],
            'selected_type_name': ['Oracle'],
            'target_type': [],
            'target_type_name': []
        },
        160: {
            'action_name': 'Morph_SurveillanceMode_quick',
            'selected_type': [82],
            'selected_type_name': ['Observer'],
            'target_type': [],
            'target_type_name': []
        },
        161: {
            'action_name': 'Effect_Restore_unit',
            'selected_type': [1910],
            'selected_type_name': ['ShieldBattery'],
            'target_type': [65, 66, 67, 68, 133, 69, 71, 72, 74, 75, 1910, 311, 59, 60, 62, 63],
            'target_type_name': [
                'TwilightCouncil', 'PhotonCannon', 'Stargate', 'TemplarArchive', 'WarpGate', 'DarkShrine',
                'RoboticsFacility', 'CyberneticsCore', 'Stalker', 'HighTemplar', 'ShieldBattery', 'Adept', 'Nexus',
                'Pylon', 'Gateway', 'Forge'
            ]
        },
        164: {
            'action_name': 'UnloadAllAt_unit',
            'selected_type': [136, 81, 893, 54],
            'selected_type_name': ['WarpPrismPhasing', 'WarpPrism', 'OverlordTransport', 'Medivac'],
            'target_type': [136, 81, 893, 54],
            'target_type_name': ['WarpPrismPhasing', 'WarpPrism', 'OverlordTransport', 'Medivac']
        },
        166: {
            'action_name': 'Train_Disruptor_quick',
            'selected_type': [71],
            'selected_type_name': ['RoboticsFacility'],
            'target_type': [],
            'target_type_name': []
        },
        167: {
            'action_name': 'Effect_PurificationNova_pt',
            'selected_type': [694],
            'selected_type_name': ['Disruptor'],
            'target_type': [],
            'target_type_name': []
        },
        168: {
            'action_name': 'raw_move_camera',
            'selected_type': [],
            'selected_type_name': [],
            'target_type': [],
            'target_type_name': []
        },
        169: {
            'action_name': 'Behavior_CloakOff_quick',
            'selected_type': [144, 50, 55],
            'selected_type_name': ['GhostAlternate', 'Ghost', 'Banshee'],
            'target_type': [],
            'target_type_name': []
        },
        172: {
            'action_name': 'Behavior_CloakOn_quick',
            'selected_type': [144, 145, 50, 55],
            'selected_type_name': ['GhostAlternate', 'GhostNova', 'Ghost', 'Banshee'],
            'target_type': [],
            'target_type_name': []
        },
        175: {
            'action_name': 'Behavior_GenerateCreepOff_quick',
            'selected_type': [106, 893],
            'selected_type_name': ['Overlord', 'OverlordTransport'],
            'target_type': [],
            'target_type_name': []
        },
        176: {
            'action_name': 'Behavior_GenerateCreepOn_quick',
            'selected_type': [106, 893],
            'selected_type_name': ['Overlord', 'OverlordTransport'],
            'target_type': [],
            'target_type_name': []
        },
        177: {
            'action_name': 'Behavior_HoldFireOff_quick',
            'selected_type': [144, 50, 503],
            'selected_type_name': ['GhostAlternate', 'Ghost', 'LurkerBurrowed'],
            'target_type': [],
            'target_type_name': []
        },
        180: {
            'action_name': 'Behavior_HoldFireOn_quick',
            'selected_type': [144, 50, 503],
            'selected_type_name': ['GhostAlternate', 'Ghost', 'LurkerBurrowed'],
            'target_type': [],
            'target_type_name': []
        },
        183: {
            'action_name': 'Build_Armory_pt',
            'selected_type': [45],
            'selected_type_name': ['SCV'],
            'target_type': [],
            'target_type_name': []
        },
        184: {
            'action_name': 'Build_BanelingNest_pt',
            'selected_type': [104],
            'selected_type_name': ['Drone'],
            'target_type': [],
            'target_type_name': []
        },
        185: {
            'action_name': 'Build_Barracks_pt',
            'selected_type': [45],
            'selected_type_name': ['SCV'],
            'target_type': [],
            'target_type_name': []
        },
        186: {
            'action_name': 'Build_Bunker_pt',
            'selected_type': [45],
            'selected_type_name': ['SCV'],
            'target_type': [],
            'target_type_name': []
        },
        187: {
            'action_name': 'Build_CommandCenter_pt',
            'selected_type': [45],
            'selected_type_name': ['SCV'],
            'target_type': [],
            'target_type_name': []
        },
        188: {
            'action_name': 'Build_CreepTumor_pt',
            'selected_type': [137, 126],
            'selected_type_name': ['CreepTumorBurrowed', 'Queen'],
            'target_type': [],
            'target_type_name': []
        },
        191: {
            'action_name': 'Build_EngineeringBay_pt',
            'selected_type': [45],
            'selected_type_name': ['SCV'],
            'target_type': [],
            'target_type_name': []
        },
        192: {
            'action_name': 'Build_EvolutionChamber_pt',
            'selected_type': [104],
            'selected_type_name': ['Drone'],
            'target_type': [],
            'target_type_name': []
        },
        193: {
            'action_name': 'Build_Extractor_unit',
            'selected_type': [104],
            'selected_type_name': ['Drone'],
            'target_type': [880, 881, 342, 343],
            'target_type_name': [
                'PurifierVespeneGeyser', 'ShakurasVespeneGeyser', 'VespeneGeyser', 'SpacePlatformGeyser'
            ]
        },
        194: {
            'action_name': 'Build_Factory_pt',
            'selected_type': [45],
            'selected_type_name': ['SCV'],
            'target_type': [],
            'target_type_name': []
        },
        195: {
            'action_name': 'Build_FusionCore_pt',
            'selected_type': [45],
            'selected_type_name': ['SCV'],
            'target_type': [],
            'target_type_name': []
        },
        196: {
            'action_name': 'Build_GhostAcademy_pt',
            'selected_type': [45],
            'selected_type_name': ['SCV'],
            'target_type': [],
            'target_type_name': []
        },
        197: {
            'action_name': 'Build_Hatchery_pt',
            'selected_type': [104],
            'selected_type_name': ['Drone'],
            'target_type': [],
            'target_type_name': []
        },
        198: {
            'action_name': 'Build_HydraliskDen_pt',
            'selected_type': [104],
            'selected_type_name': ['Drone'],
            'target_type': [],
            'target_type_name': []
        },
        199: {
            'action_name': 'Build_InfestationPit_pt',
            'selected_type': [104],
            'selected_type_name': ['Drone'],
            'target_type': [],
            'target_type_name': []
        },
        200: {
            'action_name': 'Build_Interceptors_autocast',
            'selected_type': [79],
            'selected_type_name': ['Carrier'],
            'target_type': [],
            'target_type_name': []
        },
        201: {
            'action_name': 'Build_LurkerDen_pt',
            'selected_type': [104],
            'selected_type_name': ['Drone'],
            'target_type': [],
            'target_type_name': []
        },
        202: {
            'action_name': 'Build_MissileTurret_pt',
            'selected_type': [45],
            'selected_type_name': ['SCV'],
            'target_type': [],
            'target_type_name': []
        },
        203: {
            'action_name': 'Build_Nuke_quick',
            'selected_type': [26],
            'selected_type_name': ['GhostAcademy'],
            'target_type': [],
            'target_type_name': []
        },
        204: {
            'action_name': 'Build_NydusNetwork_pt',
            'selected_type': [104],
            'selected_type_name': ['Drone'],
            'target_type': [],
            'target_type_name': []
        },
        205: {
            'action_name': 'Build_NydusWorm_pt',
            'selected_type': [95],
            'selected_type_name': ['NydusNetwork'],
            'target_type': [],
            'target_type_name': []
        },
        206: {
            'action_name': 'Build_Reactor_quick',
            'selected_type': [43, 44, 46, 21, 27, 28],
            'selected_type_name': [
                'FactoryFlying', 'StarportFlying', 'BarracksFlying', 'Barracks', 'Factory', 'Starport'
            ],
            'target_type': [],
            'target_type_name': []
        },
        207: {
            'action_name': 'Build_Reactor_pt',
            'selected_type': [43, 44, 46, 21, 27, 28],
            'selected_type_name': [
                'FactoryFlying', 'StarportFlying', 'BarracksFlying', 'Barracks', 'Factory', 'Starport'
            ],
            'target_type': [],
            'target_type_name': []
        },
        214: {
            'action_name': 'Build_Refinery_pt',
            'selected_type': [45],
            'selected_type_name': ['SCV'],
            'target_type': [880, 881, 342, 343],
            'target_type_name': [
                'PurifierVespeneGeyser', 'ShakurasVespeneGeyser', 'VespeneGeyser', 'SpacePlatformGeyser'
            ]
        },
        215: {
            'action_name': 'Build_RoachWarren_pt',
            'selected_type': [104],
            'selected_type_name': ['Drone'],
            'target_type': [],
            'target_type_name': []
        },
        216: {
            'action_name': 'Build_SensorTower_pt',
            'selected_type': [45],
            'selected_type_name': ['SCV'],
            'target_type': [],
            'target_type_name': []
        },
        217: {
            'action_name': 'Build_SpawningPool_pt',
            'selected_type': [104],
            'selected_type_name': ['Drone'],
            'target_type': [],
            'target_type_name': []
        },
        218: {
            'action_name': 'Build_SpineCrawler_pt',
            'selected_type': [104],
            'selected_type_name': ['Drone'],
            'target_type': [],
            'target_type_name': []
        },
        219: {
            'action_name': 'Build_Spire_pt',
            'selected_type': [104],
            'selected_type_name': ['Drone'],
            'target_type': [],
            'target_type_name': []
        },
        220: {
            'action_name': 'Build_SporeCrawler_pt',
            'selected_type': [104],
            'selected_type_name': ['Drone'],
            'target_type': [],
            'target_type_name': []
        },
        221: {
            'action_name': 'Build_Starport_pt',
            'selected_type': [45],
            'selected_type_name': ['SCV'],
            'target_type': [],
            'target_type_name': []
        },
        222: {
            'action_name': 'Build_SupplyDepot_pt',
            'selected_type': [45],
            'selected_type_name': ['SCV'],
            'target_type': [],
            'target_type_name': []
        },
        223: {
            'action_name': 'Build_TechLab_quick',
            'selected_type': [43, 44, 46, 21, 27, 28],
            'selected_type_name': [
                'FactoryFlying', 'StarportFlying', 'BarracksFlying', 'Barracks', 'Factory', 'Starport'
            ],
            'target_type': [],
            'target_type_name': []
        },
        224: {
            'action_name': 'Build_TechLab_pt',
            'selected_type': [43, 44, 46, 21, 27, 28],
            'selected_type_name': [
                'FactoryFlying', 'StarportFlying', 'BarracksFlying', 'Barracks', 'Factory', 'Starport'
            ],
            'target_type': [],
            'target_type_name': []
        },
        231: {
            'action_name': 'Build_UltraliskCavern_pt',
            'selected_type': [104],
            'selected_type_name': ['Drone'],
            'target_type': [],
            'target_type_name': []
        },
        232: {
            'action_name': 'BurrowDown_quick',
            'selected_type': [7, 104, 9, 105, 107, 109, 110, 111, 688, 494, 498, 502, 503, 126],
            'selected_type_name': [
                'InfestedTerran', 'Drone', 'Baneling', 'Zergling', 'Hydralisk', 'Ultralisk', 'Roach', 'Infestor',
                'Ravager', 'SwarmHost', 'WidowMine', 'Lurker', 'LurkerBurrowed', 'Queen'
            ],
            'target_type': [],
            'target_type_name': []
        },
        246: {
            'action_name': 'BurrowUp_quick',
            'selected_type': [131, 503, 493, 690, 115, 500, 117, 118, 119, 116, 120, 125, 127],
            'selected_type_name': [
                'UltraliskBurrowed', 'LurkerBurrowed', 'SwarmHostBurrowed', 'RavagerBurrowed', 'BanelingBurrowed',
                'WidowMineBurrowed', 'HydraliskBurrowed', 'RoachBurrowed', 'ZerglingBurrowed', 'DroneBurrowed',
                'InfestedTerranBurrowed', 'QueenBurrowed', 'InfestorBurrowed'
            ],
            'target_type': [],
            'target_type_name': []
        },
        247: {
            'action_name': 'BurrowUp_autocast',
            'selected_type': [115, 117, 119],
            'selected_type_name': ['BanelingBurrowed', 'HydraliskBurrowed', 'ZerglingBurrowed'],
            'target_type': [],
            'target_type_name': []
        },
        293: {
            'action_name': 'Effect_Abduct_unit',
            'selected_type': [499],
            'selected_type_name': ['Viper'],
            'target_type': [
                129, 4, 136, 9, 10, 141, 1911, 32, 289, 33, 35, 34, 45, 111, 688, 48, 689, 50, 51, 52, 692, 694, 311,
                55, 57, 691, 54, 53, 56, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 734, 484, 104, 105, 106, 107,
                108, 493, 494, 110, 495, 496, 112, 114, 498, 500, 499, 503, 1912, 502, 126, 127
            ],
            'target_type_name': [
                'Overseer', 'Colossus', 'WarpPrismPhasing', 'Baneling', 'Mothership', 'Archon',
                'ObserverSurveillanceMode', 'SiegeTankSieged', 'Broodling', 'SiegeTank', 'VikingFighter',
                'VikingAssault', 'SCV', 'Infestor', 'Ravager', 'Marine', 'Liberator', 'Ghost', 'Marauder', 'Thor',
                'Cyclone', 'Disruptor', 'Adept', 'Banshee', 'Battlecruiser', 'ThorHighImpactMode', 'Medivac', 'Hellion',
                'Raven', 'Zealot', 'Stalker', 'HighTemplar', 'DarkTemplar', 'Sentry', 'Phoenix', 'Carrier', 'VoidRay',
                'WarpPrism', 'Observer', 'Immortal', 'Probe', 'LiberatorAG', 'Hellbat', 'Drone', 'Zergling', 'Overlord',
                'Hydralisk', 'Mutalisk', 'SwarmHostBurrowed', 'SwarmHost', 'Roach', 'Oracle', 'Tempest', 'Corruptor',
                'BroodLord', 'WidowMine', 'WidowMineBurrowed', 'Viper', 'LurkerBurrowed', 'OverseerOversightMode',
                'Lurker', 'Queen', 'InfestorBurrowed'
            ]
        },
        294: {
            'action_name': 'Effect_AntiArmorMissile_unit',
            'selected_type': [56],
            'selected_type_name': ['Raven'],
            'target_type': [
                129, 8, 9, 45, 688, 693, 54, 55, 56, 57, 503, 484, 104, 489, 105, 106, 107, 108, 109, 494, 110, 113,
                112, 114, 111, 493, 115, 502, 501, 499, 118, 126, 127
            ],
            'target_type_name': [
                'Overseer', 'BanelingCocoon', 'Baneling', 'SCV', 'Ravager', 'LocustFlying', 'Medivac', 'Banshee',
                'Raven', 'Battlecruiser', 'LurkerBurrowed', 'Hellbat', 'Drone', 'Locust', 'Zergling', 'Overlord',
                'Hydralisk', 'Mutalisk', 'Ultralisk', 'SwarmHost', 'Roach', 'BroodLordCocoon', 'Corruptor', 'BroodLord',
                'Infestor', 'SwarmHostBurrowed', 'BanelingBurrowed', 'Lurker', 'LurkerCocoon', 'Viper', 'RoachBurrowed',
                'Queen', 'InfestorBurrowed'
            ]
        },
        295: {
            'action_name': 'Effect_AutoTurret_pt',
            'selected_type': [56],
            'selected_type_name': ['Raven'],
            'target_type': [],
            'target_type_name': []
        },
        296: {
            'action_name': 'Effect_BlindingCloud_pt',
            'selected_type': [499],
            'selected_type_name': ['Viper'],
            'target_type': [],
            'target_type_name': []
        },
        297: {
            'action_name': 'Effect_CalldownMULE_pt',
            'selected_type': [132],
            'selected_type_name': ['OrbitalCommand'],
            'target_type': [],
            'target_type_name': []
        },
        298: {
            'action_name': 'Effect_CalldownMULE_unit',
            'selected_type': [132],
            'selected_type_name': ['OrbitalCommand'],
            'target_type': [483, 146, 147, 884, 341, 885, 665, 666, 796, 797],
            'target_type_name': [
                'MineralField750', 'RichMineralField', 'RichMineralField750', 'PurifierMineralField', 'MineralField',
                'PurifierMineralField750', 'LabMineralField', 'LabMineralField750', 'PurifierRichMineralField',
                'PurifierRichMineralField750'
            ]
        },
        299: {
            'action_name': 'Effect_CausticSpray_unit',
            'selected_type': [112],
            'selected_type_name': ['Corruptor'],
            'target_type': [
                641, 130, 132, 133, 6, 134, 142, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 36, 37, 38, 39, 40, 41,
                42, 43, 44, 46, 47, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 589, 86, 88, 89, 90, 91, 92,
                474, 94, 95, 475, 98, 99, 100, 101, 102, 485, 364, 365, 373, 1910, 630, 504, 639
            ],
            'target_type_name': [
                'DestructibleRockEx1DiagonalHugeBLUR', 'PlanetaryFortress', 'OrbitalCommand', 'WarpGate', 'Reactor',
                'OrbitalCommandFlying', 'NydusCanal', 'CommandCenter', 'SupplyDepot', 'Refinery', 'Barracks',
                'EngineeringBay', 'MissileTurret', 'Bunker', 'GhostAcademy', 'Factory', 'Starport', 'Armory',
                'FusionCore', 'CommandCenterFlying', 'BarracksTechLab', 'BarracksReactor', 'FactoryTechLab',
                'FactoryReactor', 'StarportTechLab', 'StarportReactor', 'FactoryFlying', 'StarportFlying',
                'BarracksFlying', 'SupplyDepotLowered', 'Nexus', 'Pylon', 'Assimilator', 'Gateway', 'Forge',
                'FleetBeacon', 'TwilightCouncil', 'PhotonCannon', 'Stargate', 'TemplarArchive', 'DarkShrine',
                'RoboticsBay', 'RoboticsFacility', 'CyberneticsCore', 'CollapsibleTerranTowerDiagonal', 'Hatchery',
                'Extractor', 'SpawningPool', 'EvolutionChamber', 'HydraliskDen', 'Spire',
                'UnbuildablePlatesDestructible', 'InfestationPit', 'NydusNetwork', 'Debris2x2NonConjoined',
                'SpineCrawler', 'SporeCrawler', 'Lair', 'Hive', 'GreaterSpire', 'CollapsibleTerranTowerDebris',
                'DestructibleDebris4x4', 'DestructibleDebris6x6', 'DestructibleRampDiagonalHugeBLUR', 'ShieldBattery',
                'DestructibleCityDebrisHugeDiagonalBLUR', 'LurkerDen', 'DestructibleRockEx16x6'
            ]
        },
        301: {
            'action_name': 'Effect_Charge_unit',
            'selected_type': [73],
            'selected_type_name': ['Zealot'],
            'target_type': [98, 68, 73, 74, 75, 84, 86, 59, 126],
            'target_type_name': [
                'SpineCrawler', 'TemplarArchive', 'Zealot', 'Stalker', 'HighTemplar', 'Probe', 'Hatchery', 'Nexus',
                'Queen'
            ]
        },
        302: {
            'action_name': 'Effect_Charge_autocast',
            'selected_type': [73],
            'selected_type_name': ['Zealot'],
            'target_type': [],
            'target_type_name': []
        },
        303: {
            'action_name': 'Effect_Contaminate_unit',
            'selected_type': [1912, 129],
            'selected_type_name': ['OverseerOversightMode', 'Overseer'],
            'target_type': [
                130, 132, 133, 134, 18, 20, 21, 22, 23, 24, 27, 28, 29, 30, 36, 37, 38, 39, 41, 42, 47, 59, 60, 63, 64,
                65, 66, 67, 68, 70, 71, 72, 86, 88, 89, 90, 91, 92, 93, 94, 96, 97, 98, 100, 101, 1910
            ],
            'target_type_name': [
                'PlanetaryFortress', 'OrbitalCommand', 'WarpGate', 'OrbitalCommandFlying', 'CommandCenter', 'Refinery',
                'Barracks', 'EngineeringBay', 'MissileTurret', 'Bunker', 'Factory', 'Starport', 'Armory', 'FusionCore',
                'CommandCenterFlying', 'BarracksTechLab', 'BarracksReactor', 'FactoryTechLab', 'StarportTechLab',
                'StarportReactor', 'SupplyDepotLowered', 'Nexus', 'Pylon', 'Forge', 'FleetBeacon', 'TwilightCouncil',
                'PhotonCannon', 'Stargate', 'TemplarArchive', 'RoboticsBay', 'RoboticsFacility', 'CyberneticsCore',
                'Hatchery', 'Extractor', 'SpawningPool', 'EvolutionChamber', 'HydraliskDen', 'Spire', 'UltraliskCavern',
                'InfestationPit', 'BanelingNest', 'RoachWarren', 'SpineCrawler', 'Lair', 'Hive', 'ShieldBattery'
            ]
        },
        304: {
            'action_name': 'Effect_CorrosiveBile_pt',
            'selected_type': [688],
            'selected_type_name': ['Ravager'],
            'target_type': [],
            'target_type_name': []
        },
        305: {
            'action_name': 'Effect_EMP_pt',
            'selected_type': [50],
            'selected_type_name': ['Ghost'],
            'target_type': [],
            'target_type_name': []
        },
        307: {
            'action_name': 'Effect_Explode_quick',
            'selected_type': [9, 115],
            'selected_type_name': ['Baneling', 'BanelingBurrowed'],
            'target_type': [],
            'target_type_name': []
        },
        308: {
            'action_name': 'Effect_FungalGrowth_pt',
            'selected_type': [111],
            'selected_type_name': ['Infestor'],
            'target_type': [],
            'target_type_name': []
        },
        310: {
            'action_name': 'Effect_GhostSnipe_unit',
            'selected_type': [144, 145, 50],
            'selected_type_name': ['GhostAlternate', 'GhostNova', 'Ghost'],
            'target_type': [
                128, 129, 7, 8, 9, 15, 150, 688, 493, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115,
                499, 494, 502, 503, 1912, 501, 118, 126, 127
            ],
            'target_type_name': [
                'OverseerCocoon', 'Overseer', 'InfestedTerran', 'BanelingCocoon', 'Baneling', 'ChangelingMarine',
                'InfestedTerranCocoon', 'Ravager', 'SwarmHostBurrowed', 'Cocoon', 'Drone', 'Zergling', 'Overlord',
                'Hydralisk', 'Mutalisk', 'Ultralisk', 'Roach', 'Infestor', 'Corruptor', 'BroodLordCocoon', 'BroodLord',
                'BanelingBurrowed', 'Viper', 'SwarmHost', 'Lurker', 'LurkerBurrowed', 'OverseerOversightMode',
                'LurkerCocoon', 'RoachBurrowed', 'Queen', 'InfestorBurrowed'
            ]
        },
        311: {
            'action_name': 'Effect_Heal_unit',
            'selected_type': [54],
            'selected_type_name': ['Medivac'],
            'target_type': [48, 49, 51],
            'target_type_name': ['Marine', 'Reaper', 'Marauder']
        },
        312: {
            'action_name': 'Effect_Heal_autocast',
            'selected_type': [54],
            'selected_type_name': ['Medivac'],
            'target_type': [],
            'target_type_name': []
        },
        314: {
            'action_name': 'Effect_InfestedTerrans_pt',
            'selected_type': [111, 127],
            'selected_type_name': ['Infestor', 'InfestorBurrowed'],
            'target_type': [],
            'target_type_name': []
        },
        315: {
            'action_name': 'Effect_InjectLarva_unit',
            'selected_type': [126],
            'selected_type_name': ['Queen'],
            'target_type': [100, 101, 86],
            'target_type_name': ['Lair', 'Hive', 'Hatchery']
        },
        316: {
            'action_name': 'Effect_InterferenceMatrix_unit',
            'selected_type': [56],
            'selected_type_name': ['Raven'],
            'target_type': [499, 126, 111],
            'target_type_name': ['Viper', 'Queen', 'Infestor']
        },
        317: {
            'action_name': 'Effect_KD8Charge_pt',
            'selected_type': [49],
            'selected_type_name': ['Reaper'],
            'target_type': [],
            'target_type_name': []
        },
        318: {
            'action_name': 'Effect_LockOn_unit',
            'selected_type': [692],
            'selected_type_name': ['Cyclone'],
            'target_type': [
                129, 641, 137, 9, 139, 140, 142, 151, 687, 688, 86, 88, 90, 474, 91, 92, 96, 97, 98, 99, 100, 101, 103,
                104, 105, 106, 107, 108, 109, 110, 494, 112, 113, 114, 499, 111, 365, 630, 503, 1912, 373, 118, 893,
                126, 639
            ],
            'target_type_name': [
                'Overseer', 'DestructibleRockEx1DiagonalHugeBLUR', 'CreepTumorBurrowed', 'Baneling',
                'SpineCrawlerUprooted', 'SporeCrawlerUprooted', 'NydusCanal', 'Larva', 'RavagerCocoon', 'Ravager',
                'Hatchery', 'Extractor', 'EvolutionChamber', 'UnbuildablePlatesDestructible', 'HydraliskDen', 'Spire',
                'BanelingNest', 'RoachWarren', 'SpineCrawler', 'SporeCrawler', 'Lair', 'Hive', 'Cocoon', 'Drone',
                'Zergling', 'Overlord', 'Hydralisk', 'Mutalisk', 'Ultralisk', 'Roach', 'SwarmHost', 'Corruptor',
                'BroodLordCocoon', 'BroodLord', 'Viper', 'Infestor', 'DestructibleDebris6x6',
                'DestructibleCityDebrisHugeDiagonalBLUR', 'LurkerBurrowed', 'OverseerOversightMode',
                'DestructibleRampDiagonalHugeBLUR', 'RoachBurrowed', 'OverlordTransport', 'Queen',
                'DestructibleRockEx16x6'
            ]
        },
        319: {
            'action_name': 'Effect_LocustSwoop_pt',
            'selected_type': [693],
            'selected_type_name': ['LocustFlying'],
            'target_type': [],
            'target_type_name': []
        },
        320: {
            'action_name': 'Effect_MedivacIgniteAfterburners_quick',
            'selected_type': [54],
            'selected_type_name': ['Medivac'],
            'target_type': [],
            'target_type_name': []
        },
        321: {
            'action_name': 'Effect_NeuralParasite_unit',
            'selected_type': [127, 111],
            'selected_type_name': ['InfestorBurrowed', 'Infestor'],
            'target_type': [
                129, 4, 10, 141, 1911, 32, 33, 34, 35, 45, 48, 689, 688, 691, 51, 52, 53, 694, 55, 57, 56, 311, 54, 692,
                73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 734, 484, 104, 495, 496, 113, 498, 114, 500, 503, 1912
            ],
            'target_type_name': [
                'Overseer', 'Colossus', 'Mothership', 'Archon', 'ObserverSurveillanceMode', 'SiegeTankSieged',
                'SiegeTank', 'VikingAssault', 'VikingFighter', 'SCV', 'Marine', 'Liberator', 'Ravager',
                'ThorHighImpactMode', 'Marauder', 'Thor', 'Hellion', 'Disruptor', 'Banshee', 'Battlecruiser', 'Raven',
                'Adept', 'Medivac', 'Cyclone', 'Zealot', 'Stalker', 'HighTemplar', 'DarkTemplar', 'Sentry', 'Phoenix',
                'Carrier', 'VoidRay', 'WarpPrism', 'Observer', 'Immortal', 'Probe', 'LiberatorAG', 'Hellbat', 'Drone',
                'Oracle', 'Tempest', 'BroodLordCocoon', 'WidowMine', 'BroodLord', 'WidowMineBurrowed', 'LurkerBurrowed',
                'OverseerOversightMode'
            ]
        },
        322: {
            'action_name': 'Effect_NukeCalldown_pt',
            'selected_type': [144, 50],
            'selected_type_name': ['GhostAlternate', 'Ghost'],
            'target_type': [],
            'target_type_name': []
        },
        323: {
            'action_name': 'Effect_ParasiticBomb_unit',
            'selected_type': [499],
            'selected_type_name': ['Viper'],
            'target_type': [
                129, 4, 10, 34, 35, 689, 693, 54, 55, 56, 57, 78, 79, 80, 81, 82, 734, 108, 495, 496, 112, 114, 499,
                113, 893
            ],
            'target_type_name': [
                'Overseer', 'Colossus', 'Mothership', 'VikingAssault', 'VikingFighter', 'Liberator', 'LocustFlying',
                'Medivac', 'Banshee', 'Raven', 'Battlecruiser', 'Phoenix', 'Carrier', 'VoidRay', 'WarpPrism',
                'Observer', 'LiberatorAG', 'Mutalisk', 'Oracle', 'Tempest', 'Corruptor', 'BroodLord', 'Viper',
                'BroodLordCocoon', 'OverlordTransport'
            ]
        },
        324: {
            'action_name': 'Effect_Repair_autocast',
            'selected_type': [268, 45],
            'selected_type_name': ['MULE', 'SCV'],
            'target_type': [],
            'target_type_name': []
        },
        331: {
            'action_name': 'Effect_Restore_autocast',
            'selected_type': [1910],
            'selected_type_name': ['ShieldBattery'],
            'target_type': [],
            'target_type_name': []
        },
        332: {
            'action_name': 'Effect_Salvage_quick',
            'selected_type': [24],
            'selected_type_name': ['Bunker'],
            'target_type': [],
            'target_type_name': []
        },
        333: {
            'action_name': 'Effect_Scan_pt',
            'selected_type': [132],
            'selected_type_name': ['OrbitalCommand'],
            'target_type': [],
            'target_type_name': []
        },
        334: {
            'action_name': 'Effect_SpawnChangeling_quick',
            'selected_type': [1912, 129],
            'selected_type_name': ['OverseerOversightMode', 'Overseer'],
            'target_type': [],
            'target_type_name': []
        },
        335: {
            'action_name': 'Effect_SpawnLocusts_pt',
            'selected_type': [493, 494],
            'selected_type_name': ['SwarmHostBurrowed', 'SwarmHost'],
            'target_type': [],
            'target_type_name': []
        },
        337: {
            'action_name': 'Effect_Spray_pt',
            'selected_type': [104, 84, 45],
            'selected_type_name': ['Drone', 'Probe', 'SCV'],
            'target_type': [],
            'target_type_name': []
        },
        341: {
            'action_name': 'Effect_Stim_quick',
            'selected_type': [48, 24, 51],
            'selected_type_name': ['Marine', 'Bunker', 'Marauder'],
            'target_type': [],
            'target_type_name': []
        },
        346: {
            'action_name': 'Effect_SupplyDrop_unit',
            'selected_type': [132],
            'selected_type_name': ['OrbitalCommand'],
            'target_type': [19, 47],
            'target_type_name': ['SupplyDepot', 'SupplyDepotLowered']
        },
        347: {
            'action_name': 'Effect_TacticalJump_pt',
            'selected_type': [57],
            'selected_type_name': ['Battlecruiser'],
            'target_type': [],
            'target_type_name': []
        },
        348: {
            'action_name': 'Effect_TimeWarp_pt',
            'selected_type': [10],
            'selected_type_name': ['Mothership'],
            'target_type': [],
            'target_type_name': []
        },
        349: {
            'action_name': 'Effect_Transfusion_unit',
            'selected_type': [126],
            'selected_type_name': ['Queen'],
            'target_type': [
                128, 129, 131, 8, 137, 9, 139, 140, 504, 142, 893, 151, 289, 687, 688, 86, 88, 89, 90, 91, 92, 93, 94,
                95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 489, 105, 106, 107, 108, 109, 110, 494, 112, 493, 114, 499,
                113, 111, 502, 1912, 503, 118, 501, 892, 125, 126, 127
            ],
            'target_type_name': [
                'OverseerCocoon', 'Overseer', 'UltraliskBurrowed', 'BanelingCocoon', 'CreepTumorBurrowed', 'Baneling',
                'SpineCrawlerUprooted', 'SporeCrawlerUprooted', 'LurkerDen', 'NydusCanal', 'OverlordTransport', 'Larva',
                'Broodling', 'RavagerCocoon', 'Ravager', 'Hatchery', 'Extractor', 'SpawningPool', 'EvolutionChamber',
                'HydraliskDen', 'Spire', 'UltraliskCavern', 'InfestationPit', 'NydusNetwork', 'BanelingNest',
                'RoachWarren', 'SpineCrawler', 'SporeCrawler', 'Lair', 'Hive', 'GreaterSpire', 'Cocoon', 'Drone',
                'Locust', 'Zergling', 'Overlord', 'Hydralisk', 'Mutalisk', 'Ultralisk', 'Roach', 'SwarmHost',
                'Corruptor', 'SwarmHostBurrowed', 'BroodLord', 'Viper', 'BroodLordCocoon', 'Infestor', 'Lurker',
                'OverseerOversightMode', 'LurkerBurrowed', 'RoachBurrowed', 'LurkerCocoon', 'OverlordTransportCocoon',
                'QueenBurrowed', 'Queen', 'InfestorBurrowed'
            ]
        },
        350: {
            'action_name': 'Effect_ViperConsume_unit',
            'selected_type': [499],
            'selected_type_name': ['Viper'],
            'target_type': [96, 97, 98, 99, 100, 101, 102, 504, 139, 142, 86, 88, 89, 90, 91, 92, 93, 94, 95],
            'target_type_name': [
                'BanelingNest', 'RoachWarren', 'SpineCrawler', 'SporeCrawler', 'Lair', 'Hive', 'GreaterSpire',
                'LurkerDen', 'SpineCrawlerUprooted', 'NydusCanal', 'Hatchery', 'Extractor', 'SpawningPool',
                'EvolutionChamber', 'HydraliskDen', 'Spire', 'UltraliskCavern', 'InfestationPit', 'NydusNetwork'
            ]
        },
        363: {
            'action_name': 'Land_pt',
            'selected_type': [36, 134, 43, 44, 46],
            'selected_type_name': [
                'CommandCenterFlying', 'OrbitalCommandFlying', 'FactoryFlying', 'StarportFlying', 'BarracksFlying'
            ],
            'target_type': [],
            'target_type_name': []
        },
        369: {
            'action_name': 'Lift_quick',
            'selected_type': [132, 18, 21, 27, 28],
            'selected_type_name': ['OrbitalCommand', 'CommandCenter', 'Barracks', 'Factory', 'Starport'],
            'target_type': [],
            'target_type_name': []
        },
        375: {
            'action_name': 'LoadAll_quick',
            'selected_type': [18, 36, 130],
            'selected_type_name': ['CommandCenter', 'CommandCenterFlying', 'PlanetaryFortress'],
            'target_type': [],
            'target_type_name': []
        },
        383: {
            'action_name': 'Morph_BroodLord_quick',
            'selected_type': [112],
            'selected_type_name': ['Corruptor'],
            'target_type': [],
            'target_type_name': []
        },
        384: {
            'action_name': 'Morph_GreaterSpire_quick',
            'selected_type': [92],
            'selected_type_name': ['Spire'],
            'target_type': [],
            'target_type_name': []
        },
        385: {
            'action_name': 'Morph_Hellbat_quick',
            'selected_type': [53],
            'selected_type_name': ['Hellion'],
            'target_type': [],
            'target_type_name': []
        },
        386: {
            'action_name': 'Morph_Hellion_quick',
            'selected_type': [484],
            'selected_type_name': ['Hellbat'],
            'target_type': [],
            'target_type_name': []
        },
        387: {
            'action_name': 'Morph_Hive_quick',
            'selected_type': [100],
            'selected_type_name': ['Lair'],
            'target_type': [],
            'target_type_name': []
        },
        388: {
            'action_name': 'Morph_Lair_quick',
            'selected_type': [86],
            'selected_type_name': ['Hatchery'],
            'target_type': [],
            'target_type_name': []
        },
        389: {
            'action_name': 'Morph_LiberatorAAMode_quick',
            'selected_type': [734],
            'selected_type_name': ['LiberatorAG'],
            'target_type': [],
            'target_type_name': []
        },
        390: {
            'action_name': 'Morph_LiberatorAGMode_pt',
            'selected_type': [689],
            'selected_type_name': ['Liberator'],
            'target_type': [],
            'target_type_name': []
        },
        391: {
            'action_name': 'Morph_Lurker_quick',
            'selected_type': [107],
            'selected_type_name': ['Hydralisk'],
            'target_type': [],
            'target_type_name': []
        },
        394: {
            'action_name': 'Morph_OrbitalCommand_quick',
            'selected_type': [18],
            'selected_type_name': ['CommandCenter'],
            'target_type': [],
            'target_type_name': []
        },
        395: {
            'action_name': 'Morph_OverlordTransport_quick',
            'selected_type': [106],
            'selected_type_name': ['Overlord'],
            'target_type': [],
            'target_type_name': []
        },
        396: {
            'action_name': 'Morph_Overseer_quick',
            'selected_type': [106, 893],
            'selected_type_name': ['Overlord', 'OverlordTransport'],
            'target_type': [],
            'target_type_name': []
        },
        397: {
            'action_name': 'Morph_OverseerMode_quick',
            'selected_type': [1912],
            'selected_type_name': ['OverseerOversightMode'],
            'target_type': [],
            'target_type_name': []
        },
        398: {
            'action_name': 'Morph_OversightMode_quick',
            'selected_type': [129],
            'selected_type_name': ['Overseer'],
            'target_type': [],
            'target_type_name': []
        },
        399: {
            'action_name': 'Morph_PlanetaryFortress_quick',
            'selected_type': [18],
            'selected_type_name': ['CommandCenter'],
            'target_type': [],
            'target_type_name': []
        },
        400: {
            'action_name': 'Morph_Ravager_quick',
            'selected_type': [110],
            'selected_type_name': ['Roach'],
            'target_type': [],
            'target_type_name': []
        },
        401: {
            'action_name': 'Morph_Root_pt',
            'selected_type': [139, 140],
            'selected_type_name': ['SpineCrawlerUprooted', 'SporeCrawlerUprooted'],
            'target_type': [],
            'target_type_name': []
        },
        402: {
            'action_name': 'Morph_SiegeMode_quick',
            'selected_type': [33],
            'selected_type_name': ['SiegeTank'],
            'target_type': [],
            'target_type_name': []
        },
        407: {
            'action_name': 'Morph_SupplyDepot_Lower_quick',
            'selected_type': [19],
            'selected_type_name': ['SupplyDepot'],
            'target_type': [],
            'target_type_name': []
        },
        408: {
            'action_name': 'Morph_SupplyDepot_Raise_quick',
            'selected_type': [47],
            'selected_type_name': ['SupplyDepotLowered'],
            'target_type': [],
            'target_type_name': []
        },
        409: {
            'action_name': 'Morph_ThorExplosiveMode_quick',
            'selected_type': [691],
            'selected_type_name': ['ThorHighImpactMode'],
            'target_type': [],
            'target_type_name': []
        },
        410: {
            'action_name': 'Morph_ThorHighImpactMode_quick',
            'selected_type': [52],
            'selected_type_name': ['Thor'],
            'target_type': [],
            'target_type_name': []
        },
        411: {
            'action_name': 'Morph_Unsiege_quick',
            'selected_type': [32],
            'selected_type_name': ['SiegeTankSieged'],
            'target_type': [],
            'target_type_name': []
        },
        412: {
            'action_name': 'Morph_Uproot_quick',
            'selected_type': [98, 99],
            'selected_type_name': ['SpineCrawler', 'SporeCrawler'],
            'target_type': [],
            'target_type_name': []
        },
        413: {
            'action_name': 'Morph_VikingAssaultMode_quick',
            'selected_type': [35],
            'selected_type_name': ['VikingFighter'],
            'target_type': [],
            'target_type_name': []
        },
        414: {
            'action_name': 'Morph_VikingFighterMode_quick',
            'selected_type': [34],
            'selected_type_name': ['VikingAssault'],
            'target_type': [],
            'target_type_name': []
        },
        425: {
            'action_name': 'Research_AdaptiveTalons_quick',
            'selected_type': [504],
            'selected_type_name': ['LurkerDen'],
            'target_type': [],
            'target_type_name': []
        },
        426: {
            'action_name': 'Research_AdvancedBallistics_quick',
            'selected_type': [41],
            'selected_type_name': ['StarportTechLab'],
            'target_type': [],
            'target_type_name': []
        },
        427: {
            'action_name': 'Research_BansheeCloakingField_quick',
            'selected_type': [41],
            'selected_type_name': ['StarportTechLab'],
            'target_type': [],
            'target_type_name': []
        },
        428: {
            'action_name': 'Research_BansheeHyperflightRotors_quick',
            'selected_type': [41],
            'selected_type_name': ['StarportTechLab'],
            'target_type': [],
            'target_type_name': []
        },
        429: {
            'action_name': 'Research_BattlecruiserWeaponRefit_quick',
            'selected_type': [30],
            'selected_type_name': ['FusionCore'],
            'target_type': [],
            'target_type_name': []
        },
        430: {
            'action_name': 'Research_Burrow_quick',
            'selected_type': [100, 101, 86],
            'selected_type_name': ['Lair', 'Hive', 'Hatchery'],
            'target_type': [],
            'target_type_name': []
        },
        431: {
            'action_name': 'Research_CentrifugalHooks_quick',
            'selected_type': [96],
            'selected_type_name': ['BanelingNest'],
            'target_type': [],
            'target_type_name': []
        },
        432: {
            'action_name': 'Research_ChitinousPlating_quick',
            'selected_type': [93],
            'selected_type_name': ['UltraliskCavern'],
            'target_type': [],
            'target_type_name': []
        },
        433: {
            'action_name': 'Research_CombatShield_quick',
            'selected_type': [37],
            'selected_type_name': ['BarracksTechLab'],
            'target_type': [],
            'target_type_name': []
        },
        434: {
            'action_name': 'Research_ConcussiveShells_quick',
            'selected_type': [37],
            'selected_type_name': ['BarracksTechLab'],
            'target_type': [],
            'target_type_name': []
        },
        436: {
            'action_name': 'Research_DrillingClaws_quick',
            'selected_type': [39],
            'selected_type_name': ['FactoryTechLab'],
            'target_type': [],
            'target_type_name': []
        },
        437: {
            'action_name': 'Research_GlialRegeneration_quick',
            'selected_type': [97],
            'selected_type_name': ['RoachWarren'],
            'target_type': [],
            'target_type_name': []
        },
        438: {
            'action_name': 'Research_GroovedSpines_quick',
            'selected_type': [91],
            'selected_type_name': ['HydraliskDen'],
            'target_type': [],
            'target_type_name': []
        },
        439: {
            'action_name': 'Research_HiSecAutoTracking_quick',
            'selected_type': [22],
            'selected_type_name': ['EngineeringBay'],
            'target_type': [],
            'target_type_name': []
        },
        440: {
            'action_name': 'Research_HighCapacityFuelTanks_quick',
            'selected_type': [41],
            'selected_type_name': ['StarportTechLab'],
            'target_type': [],
            'target_type_name': []
        },
        441: {
            'action_name': 'Research_InfernalPreigniter_quick',
            'selected_type': [39],
            'selected_type_name': ['FactoryTechLab'],
            'target_type': [],
            'target_type_name': []
        },
        442: {
            'action_name': 'Research_MuscularAugments_quick',
            'selected_type': [91],
            'selected_type_name': ['HydraliskDen'],
            'target_type': [],
            'target_type_name': []
        },
        444: {
            'action_name': 'Research_NeuralParasite_quick',
            'selected_type': [94],
            'selected_type_name': ['InfestationPit'],
            'target_type': [],
            'target_type_name': []
        },
        445: {
            'action_name': 'Research_PathogenGlands_quick',
            'selected_type': [94],
            'selected_type_name': ['InfestationPit'],
            'target_type': [],
            'target_type_name': []
        },
        446: {
            'action_name': 'Research_PersonalCloaking_quick',
            'selected_type': [26],
            'selected_type_name': ['GhostAcademy'],
            'target_type': [],
            'target_type_name': []
        },
        447: {
            'action_name': 'Research_PneumatizedCarapace_quick',
            'selected_type': [100, 101, 86],
            'selected_type_name': ['Lair', 'Hive', 'Hatchery'],
            'target_type': [],
            'target_type_name': []
        },
        448: {
            'action_name': 'Research_RavenCorvidReactor_quick',
            'selected_type': [41],
            'selected_type_name': ['StarportTechLab'],
            'target_type': [],
            'target_type_name': []
        },
        450: {
            'action_name': 'Research_SmartServos_quick',
            'selected_type': [39],
            'selected_type_name': ['FactoryTechLab'],
            'target_type': [],
            'target_type_name': []
        },
        451: {
            'action_name': 'Research_Stimpack_quick',
            'selected_type': [37],
            'selected_type_name': ['BarracksTechLab'],
            'target_type': [],
            'target_type_name': []
        },
        452: {
            'action_name': 'Research_TerranInfantryArmor_quick',
            'selected_type': [22],
            'selected_type_name': ['EngineeringBay'],
            'target_type': [],
            'target_type_name': []
        },
        456: {
            'action_name': 'Research_TerranInfantryWeapons_quick',
            'selected_type': [22],
            'selected_type_name': ['EngineeringBay'],
            'target_type': [],
            'target_type_name': []
        },
        460: {
            'action_name': 'Research_TerranShipWeapons_quick',
            'selected_type': [29],
            'selected_type_name': ['Armory'],
            'target_type': [],
            'target_type_name': []
        },
        464: {
            'action_name': 'Research_TerranStructureArmorUpgrade_quick',
            'selected_type': [22],
            'selected_type_name': ['EngineeringBay'],
            'target_type': [],
            'target_type_name': []
        },
        465: {
            'action_name': 'Research_TerranVehicleAndShipPlating_quick',
            'selected_type': [29],
            'selected_type_name': ['Armory'],
            'target_type': [],
            'target_type_name': []
        },
        469: {
            'action_name': 'Research_TerranVehicleWeapons_quick',
            'selected_type': [29],
            'selected_type_name': ['Armory'],
            'target_type': [],
            'target_type_name': []
        },
        473: {
            'action_name': 'Research_TunnelingClaws_quick',
            'selected_type': [97],
            'selected_type_name': ['RoachWarren'],
            'target_type': [],
            'target_type_name': []
        },
        474: {
            'action_name': 'Research_ZergFlyerArmor_quick',
            'selected_type': [92, 102],
            'selected_type_name': ['Spire', 'GreaterSpire'],
            'target_type': [],
            'target_type_name': []
        },
        478: {
            'action_name': 'Research_ZergFlyerAttack_quick',
            'selected_type': [92, 102],
            'selected_type_name': ['Spire', 'GreaterSpire'],
            'target_type': [],
            'target_type_name': []
        },
        482: {
            'action_name': 'Research_ZergGroundArmor_quick',
            'selected_type': [90],
            'selected_type_name': ['EvolutionChamber'],
            'target_type': [],
            'target_type_name': []
        },
        486: {
            'action_name': 'Research_ZergMeleeWeapons_quick',
            'selected_type': [90],
            'selected_type_name': ['EvolutionChamber'],
            'target_type': [],
            'target_type_name': []
        },
        490: {
            'action_name': 'Research_ZergMissileWeapons_quick',
            'selected_type': [90],
            'selected_type_name': ['EvolutionChamber'],
            'target_type': [],
            'target_type_name': []
        },
        494: {
            'action_name': 'Research_ZerglingAdrenalGlands_quick',
            'selected_type': [89],
            'selected_type_name': ['SpawningPool'],
            'target_type': [],
            'target_type_name': []
        },
        495: {
            'action_name': 'Research_ZerglingMetabolicBoost_quick',
            'selected_type': [89],
            'selected_type_name': ['SpawningPool'],
            'target_type': [],
            'target_type_name': []
        },
        498: {
            'action_name': 'Train_Baneling_quick',
            'selected_type': [105],
            'selected_type_name': ['Zergling'],
            'target_type': [],
            'target_type_name': []
        },
        499: {
            'action_name': 'Train_Banshee_quick',
            'selected_type': [28],
            'selected_type_name': ['Starport'],
            'target_type': [],
            'target_type_name': []
        },
        500: {
            'action_name': 'Train_Battlecruiser_quick',
            'selected_type': [28],
            'selected_type_name': ['Starport'],
            'target_type': [],
            'target_type_name': []
        },
        501: {
            'action_name': 'Train_Corruptor_quick',
            'selected_type': [151],
            'selected_type_name': ['Larva'],
            'target_type': [],
            'target_type_name': []
        },
        502: {
            'action_name': 'Train_Cyclone_quick',
            'selected_type': [27],
            'selected_type_name': ['Factory'],
            'target_type': [],
            'target_type_name': []
        },
        503: {
            'action_name': 'Train_Drone_quick',
            'selected_type': [151],
            'selected_type_name': ['Larva'],
            'target_type': [],
            'target_type_name': []
        },
        504: {
            'action_name': 'Train_Ghost_quick',
            'selected_type': [21],
            'selected_type_name': ['Barracks'],
            'target_type': [],
            'target_type_name': []
        },
        505: {
            'action_name': 'Train_Hellbat_quick',
            'selected_type': [27],
            'selected_type_name': ['Factory'],
            'target_type': [],
            'target_type_name': []
        },
        506: {
            'action_name': 'Train_Hellion_quick',
            'selected_type': [27],
            'selected_type_name': ['Factory'],
            'target_type': [],
            'target_type_name': []
        },
        507: {
            'action_name': 'Train_Hydralisk_quick',
            'selected_type': [151],
            'selected_type_name': ['Larva'],
            'target_type': [],
            'target_type_name': []
        },
        508: {
            'action_name': 'Train_Infestor_quick',
            'selected_type': [151],
            'selected_type_name': ['Larva'],
            'target_type': [],
            'target_type_name': []
        },
        509: {
            'action_name': 'Train_Liberator_quick',
            'selected_type': [28],
            'selected_type_name': ['Starport'],
            'target_type': [],
            'target_type_name': []
        },
        510: {
            'action_name': 'Train_Marauder_quick',
            'selected_type': [21],
            'selected_type_name': ['Barracks'],
            'target_type': [],
            'target_type_name': []
        },
        511: {
            'action_name': 'Train_Marine_quick',
            'selected_type': [21],
            'selected_type_name': ['Barracks'],
            'target_type': [],
            'target_type_name': []
        },
        512: {
            'action_name': 'Train_Medivac_quick',
            'selected_type': [28],
            'selected_type_name': ['Starport'],
            'target_type': [],
            'target_type_name': []
        },
        514: {
            'action_name': 'Train_Mutalisk_quick',
            'selected_type': [151],
            'selected_type_name': ['Larva'],
            'target_type': [],
            'target_type_name': []
        },
        515: {
            'action_name': 'Train_Overlord_quick',
            'selected_type': [151],
            'selected_type_name': ['Larva'],
            'target_type': [],
            'target_type_name': []
        },
        516: {
            'action_name': 'Train_Queen_quick',
            'selected_type': [100, 101, 86],
            'selected_type_name': ['Lair', 'Hive', 'Hatchery'],
            'target_type': [],
            'target_type_name': []
        },
        517: {
            'action_name': 'Train_Raven_quick',
            'selected_type': [28],
            'selected_type_name': ['Starport'],
            'target_type': [],
            'target_type_name': []
        },
        518: {
            'action_name': 'Train_Reaper_quick',
            'selected_type': [21],
            'selected_type_name': ['Barracks'],
            'target_type': [],
            'target_type_name': []
        },
        519: {
            'action_name': 'Train_Roach_quick',
            'selected_type': [151],
            'selected_type_name': ['Larva'],
            'target_type': [],
            'target_type_name': []
        },
        520: {
            'action_name': 'Train_SCV_quick',
            'selected_type': [18, 132, 130],
            'selected_type_name': ['CommandCenter', 'OrbitalCommand', 'PlanetaryFortress'],
            'target_type': [],
            'target_type_name': []
        },
        521: {
            'action_name': 'Train_SiegeTank_quick',
            'selected_type': [27],
            'selected_type_name': ['Factory'],
            'target_type': [],
            'target_type_name': []
        },
        522: {
            'action_name': 'Train_SwarmHost_quick',
            'selected_type': [151],
            'selected_type_name': ['Larva'],
            'target_type': [],
            'target_type_name': []
        },
        523: {
            'action_name': 'Train_Thor_quick',
            'selected_type': [27],
            'selected_type_name': ['Factory'],
            'target_type': [],
            'target_type_name': []
        },
        524: {
            'action_name': 'Train_Ultralisk_quick',
            'selected_type': [151],
            'selected_type_name': ['Larva'],
            'target_type': [],
            'target_type_name': []
        },
        525: {
            'action_name': 'Train_VikingFighter_quick',
            'selected_type': [28],
            'selected_type_name': ['Starport'],
            'target_type': [],
            'target_type_name': []
        },
        526: {
            'action_name': 'Train_Viper_quick',
            'selected_type': [151],
            'selected_type_name': ['Larva'],
            'target_type': [],
            'target_type_name': []
        },
        527: {
            'action_name': 'Train_WidowMine_quick',
            'selected_type': [27],
            'selected_type_name': ['Factory'],
            'target_type': [],
            'target_type_name': []
        },
        528: {
            'action_name': 'Train_Zergling_quick',
            'selected_type': [151],
            'selected_type_name': ['Larva'],
            'target_type': [],
            'target_type_name': []
        },
        537: {
            'action_name': 'Effect_YamatoGun_unit',
            'selected_type': [57],
            'selected_type_name': ['Battlecruiser'],
            'target_type': [
                129, 130, 132, 8, 9, 137, 139, 140, 504, 142, 20, 21, 23, 151, 28, 33, 289, 34, 45, 47, 688, 503, 690,
                52, 692, 57, 472, 86, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106,
                107, 108, 109, 110, 111, 112, 113, 114, 365, 493, 373, 118, 494, 1912, 499, 501, 502, 893, 126, 127
            ],
            'target_type_name': [
                'Overseer', 'PlanetaryFortress', 'OrbitalCommand', 'BanelingCocoon', 'Baneling', 'CreepTumorBurrowed',
                'SpineCrawlerUprooted', 'SporeCrawlerUprooted', 'LurkerDen', 'NydusCanal', 'Refinery', 'Barracks',
                'MissileTurret', 'Larva', 'Starport', 'SiegeTank', 'Broodling', 'VikingAssault', 'SCV',
                'SupplyDepotLowered', 'Ravager', 'LurkerBurrowed', 'RavagerBurrowed', 'Thor', 'Cyclone',
                'Battlecruiser', 'UnbuildableRocksDestructible', 'Hatchery', 'Extractor', 'SpawningPool',
                'EvolutionChamber', 'HydraliskDen', 'Spire', 'UltraliskCavern', 'InfestationPit', 'NydusNetwork',
                'BanelingNest', 'RoachWarren', 'SpineCrawler', 'SporeCrawler', 'Lair', 'Hive', 'GreaterSpire', 'Cocoon',
                'Drone', 'Zergling', 'Overlord', 'Hydralisk', 'Mutalisk', 'Ultralisk', 'Roach', 'Infestor', 'Corruptor',
                'BroodLordCocoon', 'BroodLord', 'DestructibleDebris6x6', 'SwarmHostBurrowed',
                'DestructibleRampDiagonalHugeBLUR', 'RoachBurrowed', 'SwarmHost', 'OverseerOversightMode', 'Viper',
                'LurkerCocoon', 'Lurker', 'OverlordTransport', 'Queen', 'InfestorBurrowed'
            ]
        },
        538: {
            'action_name': 'Effect_KD8Charge_unit',
            'selected_type': [49],
            'selected_type_name': ['Reaper'],
            'target_type': [
                6, 8, 137, 138, 9, 139, 18, 19, 149, 21, 151, 24, 665, 666, 28, 289, 37, 40, 45, 687, 688, 49, 48, 47,
                692, 53, 341, 86, 343, 472, 88, 87, 89, 92, 474, 342, 96, 97, 98, 99, 100, 483, 484, 103, 104, 105, 107,
                110, 880, 881, 884, 885, 126
            ],
            'target_type_name': [
                'Reactor', 'BanelingCocoon', 'CreepTumorBurrowed', 'CreepTumorQueen', 'Baneling',
                'SpineCrawlerUprooted', 'CommandCenter', 'SupplyDepot', 'XelNagaTower', 'Barracks', 'Larva', 'Bunker',
                'LabMineralField', 'LabMineralField750', 'Starport', 'Broodling', 'BarracksTechLab', 'FactoryReactor',
                'SCV', 'RavagerCocoon', 'Ravager', 'Reaper', 'Marine', 'SupplyDepotLowered', 'Cyclone', 'Hellion',
                'MineralField', 'Hatchery', 'SpacePlatformGeyser', 'UnbuildableRocksDestructible', 'Extractor',
                'CreepTumor', 'SpawningPool', 'Spire', 'UnbuildablePlatesDestructible', 'VespeneGeyser', 'BanelingNest',
                'RoachWarren', 'SpineCrawler', 'SporeCrawler', 'Lair', 'MineralField750', 'Hellbat', 'Cocoon', 'Drone',
                'Zergling', 'Hydralisk', 'Roach', 'PurifierVespeneGeyser', 'ShakurasVespeneGeyser',
                'PurifierMineralField', 'PurifierMineralField750', 'Queen'
            ]
        },
        541: {
            'action_name': 'Effect_LockOn_autocast',
            'selected_type': [692],
            'selected_type_name': ['Cyclone'],
            'target_type': [],
            'target_type_name': []
        },
        553: {
            'action_name': 'Research_AnabolicSynthesis_quick',
            'selected_type': [93],
            'selected_type_name': ['UltraliskCavern'],
            'target_type': [],
            'target_type_name': []
        },
        554: {
            'action_name': 'Research_CycloneLockOnDamage_quick',
            'selected_type': [39],
            'selected_type_name': ['FactoryTechLab'],
            'target_type': [],
            'target_type_name': []
        },
        556: {
            'action_name': 'UnloadUnit_quick',
            'selected_type': [130, 36, 136, 81, 18, 54, 24, 893, 95],
            'selected_type_name': [
                'PlanetaryFortress', 'CommandCenterFlying', 'WarpPrismPhasing', 'WarpPrism', 'CommandCenter', 'Medivac',
                'Bunker', 'OverlordTransport', 'NydusNetwork'
            ],
            'target_type': [],
            'target_type_name': []
        },
    }

    def merge_dict(d1, d2):
        key_names = ['action_name', 'selected_type', 'selected_type_name', 'target_type', 'target_type_name']
        assert d1['action_name'] == d2['action_name']
        temp_selected_1 = {k: v for k, v in zip(d1['selected_type'], d1['selected_type_name'])}
        temp_selected_2 = {k: v for k, v in zip(d2['selected_type'], d2['selected_type_name'])}
        for k, v in temp_selected_1.items():
            if k in temp_selected_2:
                assert temp_selected_1[k] == temp_selected_2[k]
        result_selected = {**temp_selected_1, **temp_selected_2}

        temp_target_1 = {k: v for k, v in zip(d1['target_type'], d1['target_type_name'])}
        temp_target_2 = {k: v for k, v in zip(d2['target_type'], d2['target_type_name'])}
        for k, v in temp_target_1.items():
            if k in temp_target_2:
                assert temp_target_1[k] == temp_target_2[k]
        result_target = {**temp_target_1, **temp_target_2}
        result = {
            'action_name': d1['action_name'],
            'selected_type': list(result_selected.keys()),
            'selected_type_name': list(result_selected.values()),
            'target_type': list(result_target.keys()),
            'target_type_name': list(result_target.values())
        }
        return result

    result_new = {}

    for k, v in ACTIONS_STAT.items():
        if k in ACTIONS_STAT_NEW:
            result_new[k] = merge_dict(ACTIONS_STAT[k], ACTIONS_STAT_NEW[k])
        else:
            result_new[k] = v
    for k, v in ACTIONS_STAT_NEW.items():
        if k not in result_new:
            result_new[k] = v

    def print_dict(d):
        s = '{\n'
        for k, v in d.items():
            s += "    " + str(k) + ": {'action_name': '" + str(v['action_name']) + "'"
            s += ", 'selected_type': " + str(v['selected_type'])
            s += ", 'selected_type_name': " + str(v['selected_type_name'])
            s += ", 'target_type': " + str(v['target_type'])
            s += ", 'target_type_name': " + str(v['target_type_name'])
            s += "},\n"
        s += '}'
        return s

    # print(result_new)

    # s = '{\n'
    # b = {}
    # a = load_obj('/mnt/lustre/zhangming/data/stat_info.pkl')
    # x = list(a.keys())
    # x.sort()
    # total_dict = merge_units()
    # for k in x:
    #     v = a[k]
    #     list_selected_type = list(v[0])
    #     list_target_type = list(v[1])
    #     action_name = ACTION_INFO_MASK[int(k)]['name']
    #     list_selected_type_name = [total_dict[int(xx)] for xx in list_selected_type]
    #     list_target_type_name = [total_dict[int(xx)] for xx in list_target_type]

    #     b[k] = {'selected_type': list(v[0]), 'target_type': list(v[1])}
    #     # s += "    "+str(k)+": {'selected_type': "+str(list(v[0]))+", 'target_type': "+str(list(v[1]))+"},\n"
    #     s += "    "+str(k)+": {'action_name': '"+action_name+"'"
    #     s += ", 'selected_type': "+str(list_selected_type)
    #     s += ", 'selected_type_name': "+str(list_selected_type_name)
    #     s += ", 'target_type': "+str(list_target_type)
    #     s += ", 'target_type_name': "+str(list_target_type_name)
    #     s += "},\n"
    # s += '}'
    # print(s)

    with open('/mnt/lustre/zhangming/data/stat_info.txt', 'w') as f:
        s = print_dict(result_new)
        print(s)
        f.write(s)
