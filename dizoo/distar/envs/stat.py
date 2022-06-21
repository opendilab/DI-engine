from collections import defaultdict
import torch
from .static_data import ACTIONS


class Stat(object):

    def __init__(self, race_id):
        self._unit_num = defaultdict(int)
        self._unit_num['max_unit_num'] = 0
        self._race_id = race_id
        for k, v in unit_dict[race_id].items():
            self._unit_num[v] = 0
        self._action_success_count = defaultdict(int)

    def update(self, last_action_type, action_result, observation, game_step):
        if action_result < 1:
            return
        if action_result == 1:
            self.count_unit_num(last_action_type)
        entity_info, entity_num = observation['entity_info'], observation['entity_num']
        try:
            if (entity_info['alliance'][:entity_num] == 1).sum() > 10:
                self.success_rate_calc(last_action_type, action_result)
        except Exception as e:
            print('ERROR_ stat.py', e, entity_info['alliance'], entity_num)

    def success_rate_calc(self, last_action_type, action_result):
        action_name = ACTIONS[last_action_type]['name']
        error_msg = action_result_dict[action_result]
        self._action_success_count['rate/{}/{}'.format(action_name, error_msg)] += 1
        self._action_success_count['rate/{}/{}'.format(action_name, 'count')] += 1

    def get_stat_data(self):
        data = {}
        for k, v in self._unit_num.items():
            if k != 'max_unit_num':
                data['units/' + k] = v / self._unit_num['max_unit_num']
        for k, v in self._action_success_count.items():
            action_type = k.split('rate/')[1].split('/')[0]
            if 'count' in k:
                data[k] = v
            else:
                data[k] = v / (self._action_success_count['rate/{}/{}'.format(action_type, 'count')] + 1e-6)
        return data

    def count_unit_num(self, last_action_type):
        unit_name = self.get_build_unit_name(last_action_type, self._race_id)
        if not unit_name:
            return
        self._unit_num[unit_name] += 1
        self._unit_num['max_unit_num'] = max(self._unit_num[unit_name], self._unit_num['max_unit_num'])

    @staticmethod
    def get_build_unit_name(action_type, race_id):
        action_type = ACTIONS[action_type]['func_id']
        unit_name = unit_dict[race_id].get(action_type, False)
        return unit_name

    def set_race_id(self, race_id: int):
        self._race_id = race_id

    @property
    def unit_num(self):
        return self._unit_num


unit_dict = {
    'zerg': {
        383: 'BroodLord',
        391: 'Lurker',
        395: 'OverlordTransport',
        396: 'Overseer',
        400: 'Ravager',
        498: 'Baneling',
        501: 'Corruptor',
        503: 'Drone',
        507: 'Hydralisk',
        508: 'Infestor',
        514: 'Mutalisk',
        515: 'Overlord',
        516: 'Queen',
        519: 'Roach',
        522: 'SwarmHost',
        524: 'Ultralisk',
        526: 'Viper',
        528: 'Zergling'
    },
    'terran': {
        499: 'Banshee',
        500: 'Battlecruiser',
        502: 'Cyclone',
        504: 'Ghost',
        505: 'Hellbat',
        506: 'Hellion',
        509: 'Liberator',
        510: 'Marauder',
        511: 'Marine',
        512: 'Medivac',
        517: 'Raven',
        518: 'Reaper',
        520: 'SCV',
        521: 'SiegeTank',
        523: 'Thor',
        525: 'VikingFighter',
        527: 'WidowMine'
    },
    'protoss': {
        86: 'Archon',
        393: 'Mothership',
        54: 'Adept',
        56: 'Carrier',
        62: 'Colossus',
        52: 'DarkTemplar',
        166: 'Disruptor',
        51: 'HighTemplar',
        63: 'Immortal',
        513: 'MothershipCore',
        21: 'Mothership',
        61: 'Observer',
        58: 'Oracle',
        55: 'Phoenix',
        64: 'Probe',
        53: 'Sentry',
        50: 'Stalker',
        59: 'Tempest',
        57: 'VoidRay',
        76: 'Adept',
        74: 'DarkTemplar',
        73: 'HighTemplar',
        60: 'WarpPrism',
        75: 'Sentry',
        72: 'Stalker',
        71: 'Zealot',
        49: 'Zealot'
    }
}

cum_dict = [
    {
        'race': ['zerg', 'terran', 'protoss'],
        'name': 'no_op'
    }, {
        'race': ['terran'],
        'name': 'Armory'
    }, {
        'race': ['protoss'],
        'name': 'Assimilator'
    }, {
        'race': ['zerg'],
        'name': 'BanelingNest'
    }, {
        'race': ['terran'],
        'name': 'Barracks'
    }, {
        'race': ['terran'],
        'name': 'CommandCenter'
    }, {
        'race': ['protoss'],
        'name': 'CyberneticsCore'
    }, {
        'race': ['protoss'],
        'name': 'DarkShrine'
    }, {
        'race': ['terran'],
        'name': 'EngineeringBay'
    }, {
        'race': ['zerg'],
        'name': 'EvolutionChamber'
    }, {
        'race': ['zerg'],
        'name': 'Extractor'
    }, {
        'race': ['terran'],
        'name': 'Factory'
    }, {
        'race': ['protoss'],
        'name': 'FleetBeacon'
    }, {
        'race': ['protoss'],
        'name': 'Forge'
    }, {
        'race': ['terran'],
        'name': 'FusionCore'
    }, {
        'race': ['protoss'],
        'name': 'Gateway'
    }, {
        'race': ['terran'],
        'name': 'GhostAcademy'
    }, {
        'race': ['zerg'],
        'name': 'Hatchery'
    }, {
        'race': ['zerg'],
        'name': 'HydraliskDen'
    }, {
        'race': ['zerg'],
        'name': 'InfestationPit'
    }, {
        'race': ['protoss'],
        'name': 'Interceptors'
    }, {
        'race': ['protoss'],
        'name': 'Interceptors'
    }, {
        'race': ['zerg'],
        'name': 'LurkerDen'
    }, {
        'race': ['protoss'],
        'name': 'Nexus'
    }, {
        'race': ['terran'],
        'name': 'Nuke'
    }, {
        'race': ['zerg'],
        'name': 'NydusNetwork'
    }, {
        'race': ['zerg'],
        'name': 'NydusWorm'
    }, {
        'race': ['terran'],
        'name': 'Reactor'
    }, {
        'race': ['terran'],
        'name': 'Reactor'
    }, {
        'race': ['terran'],
        'name': 'Refinery'
    }, {
        'race': ['zerg'],
        'name': 'RoachWarren'
    }, {
        'race': ['protoss'],
        'name': 'RoboticsBay'
    }, {
        'race': ['protoss'],
        'name': 'RoboticsFacility'
    }, {
        'race': ['terran'],
        'name': 'SensorTower'
    }, {
        'race': ['zerg'],
        'name': 'SpawningPool'
    }, {
        'race': ['zerg'],
        'name': 'Spire'
    }, {
        'race': ['protoss'],
        'name': 'Stargate'
    }, {
        'race': ['terran'],
        'name': 'Starport'
    }, {
        'race': ['protoss'],
        'name': 'StasisTrap'
    }, {
        'race': ['terran'],
        'name': 'TechLab'
    }, {
        'race': ['terran'],
        'name': 'TechLab'
    }, {
        'race': ['protoss'],
        'name': 'TemplarArchive'
    }, {
        'race': ['protoss'],
        'name': 'TwilightCouncil'
    }, {
        'race': ['zerg'],
        'name': 'UltraliskCavern'
    }, {
        'race': ['protoss'],
        'name': 'Archon'
    }, {
        'race': ['zerg'],
        'name': 'BroodLord'
    }, {
        'race': ['zerg'],
        'name': 'GreaterSpire'
    }, {
        'race': ['zerg'],
        'name': 'Hive'
    }, {
        'race': ['zerg'],
        'name': 'Lair'
    }, {
        'race': ['zerg'],
        'name': 'LurkerDen'
    }, {
        'race': ['zerg'],
        'name': 'Lurker'
    }, {
        'race': ['protoss'],
        'name': 'Mothership'
    }, {
        'race': ['terran'],
        'name': 'OrbitalCommand'
    }, {
        'race': ['zerg'],
        'name': 'OverlordTransport'
    }, {
        'race': ['terran'],
        'name': 'PlanetaryFortress'
    }, {
        'race': ['zerg'],
        'name': 'Ravager'
    }, {
        'race': ['zerg'],
        'name': 'Research_AdaptiveTalons'
    }, {
        'race': ['protoss'],
        'name': 'Research_AdeptResonatingGlaives'
    }, {
        'race': ['terran'],
        'name': 'Research_AdvancedBallistics'
    }, {
        'race': ['zerg'],
        'name': 'Research_AnabolicSynthesis'
    }, {
        'race': ['terran'],
        'name': 'Research_BansheeCloakingField'
    }, {
        'race': ['terran'],
        'name': 'Research_BansheeHyperflightRotors'
    }, {
        'race': ['terran'],
        'name': 'Research_BattlecruiserWeaponRefit'
    }, {
        'race': ['protoss'],
        'name': 'Research_Blink'
    }, {
        'race': ['zerg'],
        'name': 'Research_Burrow'
    }, {
        'race': ['zerg'],
        'name': 'Research_CentrifugalHooks'
    }, {
        'race': ['protoss'],
        'name': 'Research_Charge'
    }, {
        'race': ['zerg'],
        'name': 'Research_ChitinousPlating'
    }, {
        'race': ['terran'],
        'name': 'Research_CombatShield'
    }, {
        'race': ['terran'],
        'name': 'Research_ConcussiveShells'
    }, {
        'race': ['terran'],
        'name': 'Research_CycloneLockOnDamage'
    }, {
        'race': ['terran'],
        'name': 'Research_CycloneRapidFireLaunchers'
    }, {
        'race': ['terran'],
        'name': 'Research_DrillingClaws'
    }, {
        'race': ['terran'],
        'name': 'Research_EnhancedShockwaves'
    }, {
        'race': ['protoss'],
        'name': 'Research_ExtendedThermalLance'
    }, {
        'race': ['zerg'],
        'name': 'Research_GlialRegeneration'
    }, {
        'race': ['protoss'],
        'name': 'Research_GraviticBooster'
    }, {
        'race': ['protoss'],
        'name': 'Research_GraviticDrive'
    }, {
        'race': ['zerg'],
        'name': 'Research_GroovedSpines'
    }, {
        'race': ['terran'],
        'name': 'Research_HighCapacityFuelTanks'
    }, {
        'race': ['terran'],
        'name': 'Research_HiSecAutoTracking'
    }, {
        'race': ['terran'],
        'name': 'Research_InfernalPreigniter'
    }, {
        'race': ['protoss'],
        'name': 'Research_InterceptorGravitonCatapult'
    }, {
        'race': ['zerg'],
        'name': 'Research_MuscularAugments'
    }, {
        'race': ['terran'],
        'name': 'Research_NeosteelFrame'
    }, {
        'race': ['zerg'],
        'name': 'Research_NeuralParasite'
    }, {
        'race': ['zerg'],
        'name': 'Research_PathogenGlands'
    }, {
        'race': ['terran'],
        'name': 'Research_PersonalCloaking'
    }, {
        'race': ['protoss'],
        'name': 'Research_PhoenixAnionPulseCrystals'
    }, {
        'race': ['zerg'],
        'name': 'Research_PneumatizedCarapace'
    }, {
        'race': ['protoss'],
        'name': 'Research_ProtossAirArmor'
    }, {
        'race': ['protoss'],
        'name': 'Research_ProtossAirWeapons'
    }, {
        'race': ['protoss'],
        'name': 'Research_ProtossGroundArmor'
    }, {
        'race': ['protoss'],
        'name': 'Research_ProtossGroundWeapons'
    }, {
        'race': ['protoss'],
        'name': 'Research_ProtossShields'
    }, {
        'race': ['protoss'],
        'name': 'Research_PsiStorm'
    }, {
        'race': ['terran'],
        'name': 'Research_RavenCorvidReactor'
    }, {
        'race': ['terran'],
        'name': 'Research_RavenRecalibratedExplosives'
    }, {
        'race': ['protoss'],
        'name': 'Research_ShadowStrike'
    }, {
        'race': ['terran'],
        'name': 'Research_SmartServos'
    }, {
        'race': ['terran'],
        'name': 'Research_Stimpack'
    }, {
        'race': ['terran'],
        'name': 'Research_TerranInfantryArmor'
    }, {
        'race': ['terran'],
        'name': 'Research_TerranInfantryWeapons'
    }, {
        'race': ['terran'],
        'name': 'Research_TerranShipWeapons'
    }, {
        'race': ['terran'],
        'name': 'Research_TerranStructureArmorUpgrade'
    }, {
        'race': ['terran'],
        'name': 'Research_TerranVehicleAndShipPlating'
    }, {
        'race': ['terran'],
        'name': 'Research_TerranVehicleWeapons'
    }, {
        'race': ['zerg'],
        'name': 'Research_TunnelingClaws'
    }, {
        'race': ['protoss'],
        'name': 'Research_WarpGate'
    }, {
        'race': ['zerg'],
        'name': 'Research_ZergFlyerArmor'
    }, {
        'race': ['zerg'],
        'name': 'Research_ZergFlyerAttack'
    }, {
        'race': ['zerg'],
        'name': 'Research_ZergGroundArmor'
    }, {
        'race': ['zerg'],
        'name': 'Research_ZerglingAdrenalGlands'
    }, {
        'race': ['zerg'],
        'name': 'Research_ZerglingMetabolicBoost'
    }, {
        'race': ['zerg'],
        'name': 'Research_ZergMeleeWeapons'
    }, {
        'race': ['zerg'],
        'name': 'Research_ZergMissileWeapons'
    }, {
        'race': ['protoss'],
        'name': 'Adept'
    }, {
        'race': ['zerg'],
        'name': 'Baneling'
    }, {
        'race': ['terran'],
        'name': 'Banshee'
    }, {
        'race': ['terran'],
        'name': 'Battlecruiser'
    }, {
        'race': ['protoss'],
        'name': 'Carrier'
    }, {
        'race': ['protoss'],
        'name': 'Colossus'
    }, {
        'race': ['zerg'],
        'name': 'Corruptor'
    }, {
        'race': ['terran'],
        'name': 'Cyclone'
    }, {
        'race': ['protoss'],
        'name': 'DarkTemplar'
    }, {
        'race': ['protoss'],
        'name': 'Disruptor'
    }, {
        'race': ['terran'],
        'name': 'Ghost'
    }, {
        'race': ['terran'],
        'name': 'Hellbat'
    }, {
        'race': ['terran'],
        'name': 'Hellion'
    }, {
        'race': ['protoss'],
        'name': 'HighTemplar'
    }, {
        'race': ['zerg'],
        'name': 'Hydralisk'
    }, {
        'race': ['protoss'],
        'name': 'Immortal'
    }, {
        'race': ['zerg'],
        'name': 'Infestor'
    }, {
        'race': ['terran'],
        'name': 'Liberator'
    }, {
        'race': ['terran'],
        'name': 'Marauder'
    }, {
        'race': ['terran'],
        'name': 'Marine'
    }, {
        'race': ['terran'],
        'name': 'Medivac'
    }, {
        'race': ['protoss'],
        'name': 'MothershipCore'
    }, {
        'race': ['protoss'],
        'name': 'Mothership'
    }, {
        'race': ['zerg'],
        'name': 'Mutalisk'
    }, {
        'race': ['protoss'],
        'name': 'Observer'
    }, {
        'race': ['protoss'],
        'name': 'Oracle'
    }, {
        'race': ['protoss'],
        'name': 'Phoenix'
    }, {
        'race': ['zerg'],
        'name': 'Queen'
    }, {
        'race': ['terran'],
        'name': 'Raven'
    }, {
        'race': ['terran'],
        'name': 'Reaper'
    }, {
        'race': ['zerg'],
        'name': 'Roach'
    }, {
        'race': ['protoss'],
        'name': 'Sentry'
    }, {
        'race': ['terran'],
        'name': 'SiegeTank'
    }, {
        'race': ['protoss'],
        'name': 'Stalker'
    }, {
        'race': ['zerg'],
        'name': 'SwarmHost'
    }, {
        'race': ['protoss'],
        'name': 'Tempest'
    }, {
        'race': ['terran'],
        'name': 'Thor'
    }, {
        'race': ['zerg'],
        'name': 'Ultralisk'
    }, {
        'race': ['terran'],
        'name': 'VikingFighter'
    }, {
        'race': ['zerg'],
        'name': 'Viper'
    }, {
        'race': ['protoss'],
        'name': 'VoidRay'
    }, {
        'race': ['protoss'],
        'name': 'Adept'
    }, {
        'race': ['protoss'],
        'name': 'DarkTemplar'
    }, {
        'race': ['protoss'],
        'name': 'HighTemplar'
    }, {
        'race': ['protoss'],
        'name': 'WarpPrism'
    }, {
        'race': ['protoss'],
        'name': 'Sentry'
    }, {
        'race': ['protoss'],
        'name': 'Stalker'
    }, {
        'race': ['protoss'],
        'name': 'Zealot'
    }, {
        'race': ['terran'],
        'name': 'WidowMine'
    }, {
        'race': ['protoss'],
        'name': 'Zealot'
    }, {
        'race': ['zerg'],
        'name': 'Zergling'
    }
]

action_result_dict = [
    '', 'Success', 'ERROR_NotSupported', 'ERROR_Error', 'ERROR_CantQueueThatOrder', 'ERROR_Retry', 'ERROR_Cooldown',
    'ERROR_QueueIsFull', 'ERROR_RallyQueueIsFull', 'ERROR_NotEnoughMinerals', 'ERROR_NotEnoughVespene',
    'ERROR_NotEnoughTerrazine', 'ERROR_NotEnoughCustom', 'ERROR_NotEnoughFood', 'ERROR_FoodUsageImpossible',
    'ERROR_NotEnoughLife', 'ERROR_NotEnoughShields', 'ERROR_NotEnoughEnergy', 'ERROR_LifeSuppressed',
    'ERROR_ShieldsSuppressed', 'ERROR_EnergySuppressed', 'ERROR_NotEnoughCharges', 'ERROR_CantAddMoreCharges',
    'ERROR_TooMuchMinerals', 'ERROR_TooMuchVespene', 'ERROR_TooMuchTerrazine', 'ERROR_TooMuchCustom',
    'ERROR_TooMuchFood', 'ERROR_TooMuchLife', 'ERROR_TooMuchShields', 'ERROR_TooMuchEnergy',
    'ERROR_MustTargetUnitWithLife', 'ERROR_MustTargetUnitWithShields', 'ERROR_MustTargetUnitWithEnergy',
    'ERROR_CantTrade', 'ERROR_CantSpend', 'ERROR_CantTargetThatUnit', 'ERROR_CouldntAllocateUnit', 'ERROR_UnitCantMove',
    'ERROR_TransportIsHoldingPosition', 'ERROR_BuildTechRequirementsNotMet', 'ERROR_CantFindPlacementLocation',
    'ERROR_CantBuildOnThat', 'ERROR_CantBuildTooCloseToDropOff', 'ERROR_CantBuildLocationInvalid',
    'ERROR_CantSeeBuildLocation', 'ERROR_CantBuildTooCloseToCreepSource', 'ERROR_CantBuildTooCloseToResources',
    'ERROR_CantBuildTooFarFromWater', 'ERROR_CantBuildTooFarFromCreepSource',
    'ERROR_CantBuildTooFarFromBuildPowerSource', 'ERROR_CantBuildOnDenseTerrain',
    'ERROR_CantTrainTooFarFromTrainPowerSource', 'ERROR_CantLandLocationInvalid', 'ERROR_CantSeeLandLocation',
    'ERROR_CantLandTooCloseToCreepSource', 'ERROR_CantLandTooCloseToResources', 'ERROR_CantLandTooFarFromWater',
    'ERROR_CantLandTooFarFromCreepSource', 'ERROR_CantLandTooFarFromBuildPowerSource',
    'ERROR_CantLandTooFarFromTrainPowerSource', 'ERROR_CantLandOnDenseTerrain', 'ERROR_AddOnTooFarFromBuilding',
    'ERROR_MustBuildRefineryFirst', 'ERROR_BuildingIsUnderConstruction', 'ERROR_CantFindDropOff',
    'ERROR_CantLoadOtherPlayersUnits', 'ERROR_NotEnoughRoomToLoadUnit', 'ERROR_CantUnloadUnitsThere',
    'ERROR_CantWarpInUnitsThere', 'ERROR_CantLoadImmobileUnits', 'ERROR_CantRechargeImmobileUnits',
    'ERROR_CantRechargeUnderConstructionUnits', 'ERROR_CantLoadThatUnit', 'ERROR_NoCargoToUnload',
    'ERROR_LoadAllNoTargetsFound', 'ERROR_NotWhileOccupied', 'ERROR_CantAttackWithoutAmmo', 'ERROR_CantHoldAnyMoreAmmo',
    'ERROR_TechRequirementsNotMet', 'ERROR_MustLockdownUnitFirst', 'ERROR_MustTargetUnit', 'ERROR_MustTargetInventory',
    'ERROR_MustTargetVisibleUnit', 'ERROR_MustTargetVisibleLocation', 'ERROR_MustTargetWalkableLocation',
    'ERROR_MustTargetPawnableUnit', 'ERROR_YouCantControlThatUnit', 'ERROR_YouCantIssueCommandsToThatUnit',
    'ERROR_MustTargetResources', 'ERROR_RequiresHealTarget', 'ERROR_RequiresRepairTarget', 'ERROR_NoItemsToDrop',
    'ERROR_CantHoldAnyMoreItems', 'ERROR_CantHoldThat', 'ERROR_TargetHasNoInventory', 'ERROR_CantDropThisItem',
    'ERROR_CantMoveThisItem', 'ERROR_CantPawnThisUnit', 'ERROR_MustTargetCaster', 'ERROR_CantTargetCaster',
    'ERROR_MustTargetOuter', 'ERROR_CantTargetOuter', 'ERROR_MustTargetYourOwnUnits', 'ERROR_CantTargetYourOwnUnits',
    'ERROR_MustTargetFriendlyUnits', 'ERROR_CantTargetFriendlyUnits', 'ERROR_MustTargetNeutralUnits',
    'ERROR_CantTargetNeutralUnits', 'ERROR_MustTargetEnemyUnits', 'ERROR_CantTargetEnemyUnits',
    'ERROR_MustTargetAirUnits', 'ERROR_CantTargetAirUnits', 'ERROR_MustTargetGroundUnits',
    'ERROR_CantTargetGroundUnits', 'ERROR_MustTargetStructures', 'ERROR_CantTargetStructures',
    'ERROR_MustTargetLightUnits', 'ERROR_CantTargetLightUnits', 'ERROR_MustTargetArmoredUnits',
    'ERROR_CantTargetArmoredUnits', 'ERROR_MustTargetBiologicalUnits', 'ERROR_CantTargetBiologicalUnits',
    'ERROR_MustTargetHeroicUnits', 'ERROR_CantTargetHeroicUnits', 'ERROR_MustTargetRoboticUnits',
    'ERROR_CantTargetRoboticUnits', 'ERROR_MustTargetMechanicalUnits', 'ERROR_CantTargetMechanicalUnits',
    'ERROR_MustTargetPsionicUnits', 'ERROR_CantTargetPsionicUnits', 'ERROR_MustTargetMassiveUnits',
    'ERROR_CantTargetMassiveUnits', 'ERROR_MustTargetMissile', 'ERROR_CantTargetMissile', 'ERROR_MustTargetWorkerUnits',
    'ERROR_CantTargetWorkerUnits', 'ERROR_MustTargetEnergyCapableUnits', 'ERROR_CantTargetEnergyCapableUnits',
    'ERROR_MustTargetShieldCapableUnits', 'ERROR_CantTargetShieldCapableUnits', 'ERROR_MustTargetFlyers',
    'ERROR_CantTargetFlyers', 'ERROR_MustTargetBuriedUnits', 'ERROR_CantTargetBuriedUnits',
    'ERROR_MustTargetCloakedUnits', 'ERROR_CantTargetCloakedUnits', 'ERROR_MustTargetUnitsInAStasisField',
    'ERROR_CantTargetUnitsInAStasisField', 'ERROR_MustTargetUnderConstructionUnits',
    'ERROR_CantTargetUnderConstructionUnits', 'ERROR_MustTargetDeadUnits', 'ERROR_CantTargetDeadUnits',
    'ERROR_MustTargetRevivableUnits', 'ERROR_CantTargetRevivableUnits', 'ERROR_MustTargetHiddenUnits',
    'ERROR_CantTargetHiddenUnits', 'ERROR_CantRechargeOtherPlayersUnits', 'ERROR_MustTargetHallucinations',
    'ERROR_CantTargetHallucinations', 'ERROR_MustTargetInvulnerableUnits', 'ERROR_CantTargetInvulnerableUnits',
    'ERROR_MustTargetDetectedUnits', 'ERROR_CantTargetDetectedUnits', 'ERROR_CantTargetUnitWithEnergy',
    'ERROR_CantTargetUnitWithShields', 'ERROR_MustTargetUncommandableUnits', 'ERROR_CantTargetUncommandableUnits',
    'ERROR_MustTargetPreventDefeatUnits', 'ERROR_CantTargetPreventDefeatUnits', 'ERROR_MustTargetPreventRevealUnits',
    'ERROR_CantTargetPreventRevealUnits', 'ERROR_MustTargetPassiveUnits', 'ERROR_CantTargetPassiveUnits',
    'ERROR_MustTargetStunnedUnits', 'ERROR_CantTargetStunnedUnits', 'ERROR_MustTargetSummonedUnits',
    'ERROR_CantTargetSummonedUnits', 'ERROR_MustTargetUser1', 'ERROR_CantTargetUser1',
    'ERROR_MustTargetUnstoppableUnits', 'ERROR_CantTargetUnstoppableUnits', 'ERROR_MustTargetResistantUnits',
    'ERROR_CantTargetResistantUnits', 'ERROR_MustTargetDazedUnits', 'ERROR_CantTargetDazedUnits', 'ERROR_CantLockdown',
    'ERROR_CantMindControl', 'ERROR_MustTargetDestructibles', 'ERROR_CantTargetDestructibles', 'ERROR_MustTargetItems',
    'ERROR_CantTargetItems', 'ERROR_NoCalldownAvailable', 'ERROR_WaypointListFull', 'ERROR_MustTargetRace',
    'ERROR_CantTargetRace', 'ERROR_MustTargetSimilarUnits', 'ERROR_CantTargetSimilarUnits',
    'ERROR_CantFindEnoughTargets', 'ERROR_AlreadySpawningLarva', 'ERROR_CantTargetExhaustedResources',
    'ERROR_CantUseMinimap', 'ERROR_CantUseInfoPanel', 'ERROR_OrderQueueIsFull', 'ERROR_CantHarvestThatResource',
    'ERROR_HarvestersNotRequired', 'ERROR_AlreadyTargeted', 'ERROR_CantAttackWeaponsDisabled',
    'ERROR_CouldntReachTarget', 'ERROR_TargetIsOutOfRange', 'ERROR_TargetIsTooClose', 'ERROR_TargetIsOutOfArc',
    'ERROR_CantFindTeleportLocation', 'ERROR_InvalidItemClass', 'ERROR_CantFindCancelOrder'
]
NUM_ACTION_RESULT = 214

ACTION_RACE_MASK = {
    'zerg': torch.tensor(
        [
            False, False, True, True, True, True, False, False, True, True, True, True, False, False, False, False,
            True, False, False, False, True, False, False, False, True, True, False, False, False, False, False, False,
            True, True, True, False, False, True, False, False, False, True, True, False, False, False, False, False,
            True, False, False, False, False, True, True, True, True, False, False, False, False, False, False, False,
            False, True, True, True, True, True, True, True, False, False, False, True, False, False, False, False,
            True, False, False, False, False, False, True, True, False, False, True, False, False, True, True, False,
            False, False, False, False, False, False, True, True, False, False, False, False, False, True, False, False,
            True, False, False, True, False, False, False, False, False, False, False, False, False, True, True, True,
            True, False, False, False, False, True, True, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False, False, True, True, True, False, False, False,
            True, False, True, False, True, False, False, True, True, False, False, True, True, False, False, False,
            True, True, True, True, False, True, True, False, False, False, False, False, False, False, True, False,
            False, False, False, False, False, True, True, True, True, True, True, True, True, True, False, False, True,
            False, False, False, False, True, True, False, True, False, False, False, False, False, False, False, True,
            False, False, True, False, False, False, False, True, False, True, True, False, False, True, False, False,
            False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
            True, False, True, True, True, True, True, True, True, True, True, True, False, True, False, False, False,
            False, True, False, False, False, True, False, False, False, False, True, False, True, False, False, False,
            False, False, False, True, False, False, True, False, False, True, False, False, True, False, False, False,
            False, True, False, False, True, False, True, False, False, False, False, False, False, False, False, False,
            False, True, True, True, True, True
        ]
    ),
    'terran': torch.tensor(
        [
            False, False, True, True, False, False, True, True, False, False, True, True, False, False, True, False,
            False, True, True, True, False, False, False, True, False, False, True, False, False, True, False, True,
            False, False, False, False, False, False, True, False, True, False, False, False, False, True, True, True,
            False, False, False, True, False, False, False, False, False, False, True, False, True, True, True, False,
            False, False, True, True, True, True, True, False, False, True, True, False, False, False, True, True,
            False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, True,
            False, False, True, True, False, False, False, False, True, True, True, True, True, False, False, True,
            False, True, False, False, False, False, True, True, True, False, False, True, True, False, False, False,
            True, True, True, True, False, False, False, False, True, True, True, True, False, False, False, False,
            False, False, False, False, False, False, False, False, False, True, True, True, True, True, True, True,
            True, False, False, False, False, True, True, False, False, True, True, False, False, False, False, True,
            False, False, False, False, True, False, False, True, True, True, False, True, True, True, False, True,
            True, False, False, False, False, True, True, True, True, True, True, True, True, False, False, True, False,
            True, True, True, False, False, False, False, False, True, True, True, True, True, True, False, False,
            False, False, False, True, True, True, False, False, True, False, False, True, False, False, False, False,
            False, False, False, False, True, True, False, True, True, True, True, True, True, True, True, False, False,
            False, False, False, False, False, False, False, True, True, True, False, False, True, True, False, False,
            False, True, False, False, False, True, True, True, False, False, False, False, True, True, True, True,
            False, False, False, False, False, False, False, False, False, True, True, False, True, False, True, False,
            False, False, True, False, True, False, False, False, False, False, False, False, False, False, True, False,
            False, True, True, True, True
        ]
    ),
    'protoss': torch.tensor(
        [
            False, False, True, True, False, False, False, False, False, False, False, False, True, True, False, True,
            False, False, False, False, False, True, True, False, False, False, False, True, True, False, True, False,
            False, False, False, True, True, False, False, True, False, False, False, True, True, False, False, False,
            False, True, True, False, True, False, False, False, False, True, False, True, False, False, False, True,
            True, False, False, False, False, True, True, False, True, False, False, False, True, True, False, False,
            False, True, True, True, True, True, False, False, False, False, False, True, True, False, False, False,
            True, True, False, False, True, True, False, False, False, False, False, False, False, False, True, False,
            False, False, True, False, True, True, False, False, False, True, True, False, False, False, False, False,
            True, False, False, False, True, False, False, True, False, False, False, False, True, True, True, True,
            True, True, True, True, True, True, True, True, True, False, True, True, True, False, False, False, True,
            True, False, True, False, False, False, False, False, False, False, False, False, True, True, False, False,
            False, False, False, False, False, False, False, False, False, True, False, False, False, False, False,
            False, True, True, True, True, True, True, True, True, True, True, True, True, False, True, False, False,
            False, False, False, True, False, False, True, False, False, False, False, False, False, False, True, False,
            True, True, False, False, False, False, True, False, False, False, False, False, True, False, True, True,
            True, True, True, True, False, False, True, False, False, False, False, False, False, False, False, False,
            True, False, False, False, False, False, False, False, True, True, True, True, False, False, False, True,
            True, False, False, True, True, False, False, False, False, True, False, True, False, False, False, False,
            False, True, True, False, True, True, False, True, True, False, False, False, False, False, True, False,
            True, False, True, False, False, False, False, True, True, True, True, True, True, True, True, False, True,
            False, True, True, False, True
        ]
    )
}
