from pysc2.lib import RACE, UNIT_TYPEID, UPGRADE_ID, ABILITY_ID 
from pysc2.lib import data_raw_3_16, data_raw_4_0
import distutils.version

class TypeData(object):
    def __init__(self, race=0, mineralCost=0, gasCost=0, supplyCost=0, buildTime=0, isUnit=False, isBuilding=False, isWorker=False, isRefinery=False, isSupplyProvider=False, isResourceDepot=False, isAddon=False, buildAbility=0, warpAbility=0, whatBuilds=[], requiredUnits=[], requiredUpgrades=[]):
        self.race = race
        self.mineralCost = mineralCost
        self.gasCost = gasCost
        self.supplyCost = supplyCost
        self.buildTime = buildTime
        self.isUnit = isUnit
        self.isBuilding = isBuilding
        self.isWorker = isWorker
        self.isRefinery = isRefinery
        self.isSupplyProvider = isSupplyProvider
        self.isResourceDepot = isResourceDepot
        self.isAddon = isAddon
        self.buildAbility = buildAbility
        self.warpAbility = warpAbility
        self.whatBuilds = whatBuilds
        self.requiredUnits = requiredUnits
        self.requiredUpgrades = requiredUpgrades

class TechTree(object):
    def __init__(self):
        self.fps = 16
        self.m_unitTypeData = {}
        self.m_upgradeData = {}
        self.initUnitTypeData()
        self.initUpgradeData()

    def update_version(self, version):
        self.updateUnitTypeData(version)
        self.updateUpgradeData(version)

    def getUnitData(self,unit_id):
        if unit_id in self.m_unitTypeData:
            return self.m_unitTypeData[unit_id]
        else:
            print('Unrecognized Unit ID %d.' % unit_id)
            raise KeyError

    def getUpgradeData(self,upgrade_id):
        if upgrade_id in self.m_upgradeData:
            return self.m_upgradeData[upgrade_id]
        else:
            print('Unrecognized Upgrade ID %d.' % upgrade_id)
            raise KeyError

    def initUnitTypeData(self):

    # Protoss Buildings                                                                                  unit  bld   wrk    rfn    sup    hall   add
        self.m_unitTypeData[UNIT_TYPEID.PROTOSS_PYLONOVERCHARGED.value] = TypeData( RACE.Protoss, 100, 0, 0, 0, True, True, False, False,  True, False, False, ABILITY_ID.EFFECT_PHOTONOVERCHARGE.value, 0, [ UNIT_TYPEID.PROTOSS_MOTHERSHIPCORE.value, UNIT_TYPEID.PROTOSS_PYLON.value ], [], [] )
        self.m_unitTypeData[UNIT_TYPEID.PROTOSS_PYLON.value] =            TypeData( RACE.Protoss, 100, 0, 0, self.fps*25, True, True, False, False,  True, False, False, ABILITY_ID.BUILD_PYLON.value, 0, [ UNIT_TYPEID.PROTOSS_PROBE.value ], [], [] )
        self.m_unitTypeData[UNIT_TYPEID.PROTOSS_NEXUS.value] =            TypeData( RACE.Protoss, 400, 0, 0, self.fps*100, True, True, False, False, False,  True, False, ABILITY_ID.BUILD_NEXUS.value, 0, [ UNIT_TYPEID.PROTOSS_PROBE.value ], [], [] )
        self.m_unitTypeData[UNIT_TYPEID.PROTOSS_ASSIMILATOR.value] =      TypeData( RACE.Protoss, 75, 0, 0, self.fps*30, True, True, False,  True, False, False, False, ABILITY_ID.BUILD_ASSIMILATOR.value, 0, [ UNIT_TYPEID.PROTOSS_PROBE.value ], [], [] )
        self.m_unitTypeData[UNIT_TYPEID.PROTOSS_CYBERNETICSCORE.value] =  TypeData( RACE.Protoss, 150, 0, 0, self.fps*50, True, True, False, False, False, False, False, ABILITY_ID.BUILD_CYBERNETICSCORE.value, 0, [ UNIT_TYPEID.PROTOSS_PROBE.value ], [ UNIT_TYPEID.PROTOSS_GATEWAY.value, UNIT_TYPEID.PROTOSS_WARPGATE.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.PROTOSS_DARKSHRINE.value] =       TypeData( RACE.Protoss, 150, 150, 0, 0, True, True, False, False, False, False, False, ABILITY_ID.BUILD_DARKSHRINE.value, 0, [ UNIT_TYPEID.PROTOSS_PROBE.value ], [ UNIT_TYPEID.PROTOSS_TWILIGHTCOUNCIL.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.PROTOSS_FLEETBEACON.value] =      TypeData( RACE.Protoss, 300, 200, 0, 0, True, True, False, False, False, False, False, ABILITY_ID.BUILD_FLEETBEACON.value, 0, [ UNIT_TYPEID.PROTOSS_PROBE.value ], [ UNIT_TYPEID.PROTOSS_STARGATE.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.PROTOSS_FORGE.value] =            TypeData( RACE.Protoss, 150, 0, 0, 0, True, True, False, False, False, False, False, ABILITY_ID.BUILD_FORGE.value, 0, [ UNIT_TYPEID.PROTOSS_PROBE.value ], [ UNIT_TYPEID.PROTOSS_NEXUS.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.PROTOSS_GATEWAY.value] =          TypeData( RACE.Protoss, 150, 0, 0, 0, True, True, False, False, False, False, False, ABILITY_ID.BUILD_GATEWAY.value, 0, [ UNIT_TYPEID.PROTOSS_PROBE.value ], [ UNIT_TYPEID.PROTOSS_NEXUS.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.PROTOSS_STARGATE.value] =         TypeData( RACE.Protoss, 150, 150, 0, 0, True, True, False, False, False, False, False, ABILITY_ID.BUILD_STARGATE.value, 0, [ UNIT_TYPEID.PROTOSS_PROBE.value ], [ UNIT_TYPEID.PROTOSS_CYBERNETICSCORE.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.PROTOSS_PHOTONCANNON.value] =     TypeData( RACE.Protoss, 150, 0, 0, 0, True, True, False, False, False, False, False, ABILITY_ID.BUILD_PHOTONCANNON.value, 0, [ UNIT_TYPEID.PROTOSS_PROBE.value ], [ UNIT_TYPEID.PROTOSS_FORGE.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.PROTOSS_ROBOTICSBAY.value] =      TypeData( RACE.Protoss, 200, 200, 0, 0, True, True, False, False, False, False, False, ABILITY_ID.BUILD_ROBOTICSBAY.value, 0, [ UNIT_TYPEID.PROTOSS_PROBE.value ], [ UNIT_TYPEID.PROTOSS_ROBOTICSFACILITY.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.PROTOSS_ROBOTICSFACILITY.value] = TypeData( RACE.Protoss, 200, 100, 0, 0, True, True, False, False, False, False, False, ABILITY_ID.BUILD_ROBOTICSFACILITY.value, 0, [ UNIT_TYPEID.PROTOSS_PROBE.value ], [ UNIT_TYPEID.PROTOSS_CYBERNETICSCORE.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.PROTOSS_TEMPLARARCHIVE.value] =   TypeData( RACE.Protoss, 150, 200, 0, 0, True, True, False, False, False, False, False, ABILITY_ID.BUILD_TEMPLARARCHIVE.value, 0, [ UNIT_TYPEID.PROTOSS_PROBE.value ], [ UNIT_TYPEID.PROTOSS_TWILIGHTCOUNCIL.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.PROTOSS_TWILIGHTCOUNCIL.value] =  TypeData( RACE.Protoss, 150, 100, 0, 0, True, True, False, False, False, False, False, ABILITY_ID.BUILD_TWILIGHTCOUNCIL.value, 0, [ UNIT_TYPEID.PROTOSS_PROBE.value ], [ UNIT_TYPEID.PROTOSS_CYBERNETICSCORE.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.PROTOSS_WARPGATE.value] =         TypeData( RACE.Protoss, 150, 0, 0, 0, True, True, False, False, False, False, False, ABILITY_ID.MORPH_WARPGATE.value, 0, [ UNIT_TYPEID.PROTOSS_GATEWAY.value ], [], [ UPGRADE_ID.WARPGATERESEARCH ] )

    # Protoss Units                                                                                      unit  bld   wrk    rfn    sup    hall   add
        self.m_unitTypeData[UNIT_TYPEID.PROTOSS_PROBE.value] =            TypeData( RACE.Protoss, 50, 0, 1, 0, True, False,  True, False, False, False, False, ABILITY_ID.TRAIN_PROBE.value, 0, [ UNIT_TYPEID.PROTOSS_NEXUS.value ], [], [] )
        self.m_unitTypeData[UNIT_TYPEID.PROTOSS_MOTHERSHIPCORE.value] =   TypeData( RACE.Protoss, 100, 100, 2, 0, True, False, False, False, False, False, False, ABILITY_ID.TRAIN_MOTHERSHIPCORE.value, 0, [ UNIT_TYPEID.PROTOSS_NEXUS.value ], [], [] )
        self.m_unitTypeData[UNIT_TYPEID.PROTOSS_ZEALOT.value] =           TypeData( RACE.Protoss, 100, 0, 2, 0, True, False, False, False, False, False, False, ABILITY_ID.TRAIN_ZEALOT.value, ABILITY_ID.TRAINWARP_ZEALOT.value, [ UNIT_TYPEID.PROTOSS_GATEWAY.value ], [], [] )
        self.m_unitTypeData[UNIT_TYPEID.PROTOSS_SENTRY.value] =           TypeData( RACE.Protoss, 50, 100, 2, 0, True, False, False, False, False, False, False, ABILITY_ID.TRAIN_SENTRY.value, ABILITY_ID.TRAINWARP_SENTRY.value, [ UNIT_TYPEID.PROTOSS_GATEWAY.value ], [ UNIT_TYPEID.PROTOSS_CYBERNETICSCORE.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.PROTOSS_STALKER.value] =          TypeData( RACE.Protoss, 125, 50, 2, 0, True, False, False, False, False, False, False, ABILITY_ID.TRAIN_STALKER.value, ABILITY_ID.TRAINWARP_STALKER.value, [ UNIT_TYPEID.PROTOSS_GATEWAY.value ], [ UNIT_TYPEID.PROTOSS_CYBERNETICSCORE.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.PROTOSS_HIGHTEMPLAR.value] =      TypeData( RACE.Protoss, 50, 150, 2, 0, True, False, False, False, False, False, False, ABILITY_ID.TRAIN_HIGHTEMPLAR.value, ABILITY_ID.TRAINWARP_HIGHTEMPLAR.value, [ UNIT_TYPEID.PROTOSS_GATEWAY.value ], [ UNIT_TYPEID.PROTOSS_TEMPLARARCHIVE.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.PROTOSS_DARKTEMPLAR.value] =      TypeData( RACE.Protoss, 125, 125, 2, 0, True, False, False, False, False, False, False, ABILITY_ID.TRAIN_DARKTEMPLAR.value, ABILITY_ID.TRAINWARP_DARKTEMPLAR.value, [ UNIT_TYPEID.PROTOSS_GATEWAY.value ], [ UNIT_TYPEID.PROTOSS_DARKSHRINE.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.PROTOSS_ADEPT.value] =            TypeData( RACE.Protoss, 100, 25, 2, 0, True, False, False, False, False, False, False, ABILITY_ID.TRAIN_ADEPT.value, ABILITY_ID.TRAINWARP_ADEPT.value, [ UNIT_TYPEID.PROTOSS_GATEWAY.value ], [ UNIT_TYPEID.PROTOSS_CYBERNETICSCORE.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.PROTOSS_COLOSSUS.value] =         TypeData( RACE.Protoss, 300, 200, 6, 0, True, False, False, False, False, False, False, ABILITY_ID.TRAIN_COLOSSUS.value,  0, [ UNIT_TYPEID.PROTOSS_ROBOTICSFACILITY.value ], [ UNIT_TYPEID.PROTOSS_ROBOTICSBAY.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.PROTOSS_DISRUPTOR.value] =        TypeData( RACE.Protoss, 150, 150, 3, 0, True, False, False, False, False, False, False, ABILITY_ID.TRAIN_DISRUPTOR.value, 0, [ UNIT_TYPEID.PROTOSS_ROBOTICSFACILITY.value ], [ UNIT_TYPEID.PROTOSS_ROBOTICSBAY.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.PROTOSS_WARPPRISM.value] =        TypeData( RACE.Protoss, 200, 0, 2, 0, True, False, False, False, False, False, False, ABILITY_ID.TRAIN_WARPPRISM.value, 0, [ UNIT_TYPEID.PROTOSS_ROBOTICSFACILITY.value ], [], [] )
        self.m_unitTypeData[UNIT_TYPEID.PROTOSS_OBSERVER.value] =         TypeData( RACE.Protoss, 25, 75, 1, 0, True, False, False, False, False, False, False, ABILITY_ID.TRAIN_OBSERVER.value, 0, [ UNIT_TYPEID.PROTOSS_ROBOTICSFACILITY.value ], [], [] )
        self.m_unitTypeData[UNIT_TYPEID.PROTOSS_IMMORTAL.value] =         TypeData( RACE.Protoss, 250, 100, 4, 0, True, False, False, False, False, False, False, ABILITY_ID.TRAIN_IMMORTAL.value, 0, [ UNIT_TYPEID.PROTOSS_ROBOTICSFACILITY.value ], [], [] )
        self.m_unitTypeData[UNIT_TYPEID.PROTOSS_CARRIER.value] =          TypeData( RACE.Protoss, 350, 250, 6, 0, True, False, False, False, False, False, False, ABILITY_ID.TRAIN_CARRIER.value, 0, [ UNIT_TYPEID.PROTOSS_STARGATE.value ], [ UNIT_TYPEID.PROTOSS_FLEETBEACON.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.PROTOSS_ORACLE.value] =           TypeData( RACE.Protoss, 150, 150, 3, 0, True, False, False, False, False, False, False, ABILITY_ID.TRAIN_ORACLE.value, 0, [ UNIT_TYPEID.PROTOSS_STARGATE.value ], [], [] )
        self.m_unitTypeData[UNIT_TYPEID.PROTOSS_PHOENIX.value] =          TypeData( RACE.Protoss, 150, 100, 2, 0, True, False, False, False, False, False, False, ABILITY_ID.TRAIN_PHOENIX.value, 0, [ UNIT_TYPEID.PROTOSS_STARGATE.value ], [], [] )
        self.m_unitTypeData[UNIT_TYPEID.PROTOSS_VOIDRAY.value] =          TypeData( RACE.Protoss, 250, 150, 4, 0, True, False, False, False, False, False, False, ABILITY_ID.TRAIN_VOIDRAY.value, 0, [ UNIT_TYPEID.PROTOSS_STARGATE.value ], [], [] )
        self.m_unitTypeData[UNIT_TYPEID.PROTOSS_TEMPEST.value] =          TypeData( RACE.Protoss, 300, 200, 4, 0, True, False, False, False, False, False, False, ABILITY_ID.TRAIN_TEMPEST.value, 0, [ UNIT_TYPEID.PROTOSS_STARGATE.value ], [ UNIT_TYPEID.PROTOSS_FLEETBEACON.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.PROTOSS_INTERCEPTOR.value] =      TypeData( RACE.Protoss, 10, 0, 0, 0, True, False, False, False, False, False, False, ABILITY_ID.BUILD_INTERCEPTORS.value, 0, [ UNIT_TYPEID.PROTOSS_CARRIER.value ], [], [] )
        self.m_unitTypeData[UNIT_TYPEID.PROTOSS_ORACLESTASISTRAP.value] = TypeData( RACE.Protoss, 0, 0, 0, 0, True, False, False, False, False, False, False, ABILITY_ID.BUILD_STASISTRAP.value, 0, [ UNIT_TYPEID.PROTOSS_ORACLE.value ], [], [] )

    # Terran Buildings                                                                          m  g  s  t  unit  bld   wrk    rfn    sup    hall   add
        self.m_unitTypeData[UNIT_TYPEID.TERRAN_COMMANDCENTER.value] =     TypeData( RACE.Terran, 400, 0, 0, 0, True, True, False, False, False,  True, False, ABILITY_ID.BUILD_COMMANDCENTER.value, 0, [ UNIT_TYPEID.TERRAN_SCV.value ], [], [] )
        self.m_unitTypeData[UNIT_TYPEID.TERRAN_SUPPLYDEPOT.value] =       TypeData( RACE.Terran, 100, 0, 0, 0, True, True, False, False,  True, False, False, ABILITY_ID.BUILD_SUPPLYDEPOT.value, 0, [ UNIT_TYPEID.TERRAN_SCV.value ], [], []  )
        self.m_unitTypeData[UNIT_TYPEID.TERRAN_REFINERY.value] =          TypeData( RACE.Terran, 75, 0, 0, 0, True, True, False,  True, False, False, False, ABILITY_ID.BUILD_REFINERY.value, 0, [ UNIT_TYPEID.TERRAN_SCV.value ], [], [] )
        self.m_unitTypeData[UNIT_TYPEID.TERRAN_ARMORY.value] =            TypeData( RACE.Terran, 150, 100, 0, 0, True, True, False, False, False, False, False, ABILITY_ID.BUILD_ARMORY.value, 0, [ UNIT_TYPEID.TERRAN_SCV.value ], [ UNIT_TYPEID.TERRAN_FACTORY.value, UNIT_TYPEID.TERRAN_FACTORYFLYING.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.TERRAN_BARRACKS.value] =          TypeData( RACE.Terran, 150, 0, 0, 0, True, True, False, False, False, False, False, ABILITY_ID.BUILD_BARRACKS.value, 0, [ UNIT_TYPEID.TERRAN_SCV.value ], [ UNIT_TYPEID.TERRAN_SUPPLYDEPOT.value, UNIT_TYPEID.TERRAN_SUPPLYDEPOTLOWERED.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.TERRAN_SENSORTOWER.value] =       TypeData( RACE.Terran, 125, 100, 0, 0, True, True, False, False, False, False, False, ABILITY_ID.BUILD_SENSORTOWER.value, 0, [ UNIT_TYPEID.TERRAN_SCV.value ], [ UNIT_TYPEID.TERRAN_ENGINEERINGBAY.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.TERRAN_FACTORY.value] =           TypeData( RACE.Terran, 150, 100, 0, 0, True, True, False, False, False, False, False, ABILITY_ID.BUILD_FACTORY.value, 0, [ UNIT_TYPEID.TERRAN_SCV.value ], [ UNIT_TYPEID.TERRAN_BARRACKS.value, UNIT_TYPEID.TERRAN_BARRACKSFLYING.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.TERRAN_FUSIONCORE.value] =        TypeData( RACE.Terran, 150, 150, 0, 0, True, True, False, False, False, False, False, ABILITY_ID.BUILD_FUSIONCORE.value, 0, [ UNIT_TYPEID.TERRAN_SCV.value ], [ UNIT_TYPEID.TERRAN_STARPORT.value, UNIT_TYPEID.TERRAN_STARPORTFLYING.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.TERRAN_STARPORT.value] =          TypeData( RACE.Terran, 150, 100, 0, 0, True, True, False, False, False, False, False, ABILITY_ID.BUILD_STARPORT.value, 0, [ UNIT_TYPEID.TERRAN_SCV.value ], [ UNIT_TYPEID.TERRAN_FACTORY.value, UNIT_TYPEID.TERRAN_FACTORYFLYING.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.TERRAN_GHOSTACADEMY.value] =      TypeData( RACE.Terran, 150, 50, 0, 0, True, True, False, False, False, False, False, ABILITY_ID.BUILD_GHOSTACADEMY.value, 0, [ UNIT_TYPEID.TERRAN_SCV.value ], [ UNIT_TYPEID.TERRAN_BARRACKS.value, UNIT_TYPEID.TERRAN_BARRACKSFLYING.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.TERRAN_BUNKER.value] =            TypeData( RACE.Terran, 100, 0, 0, 0, True, True, False, False, False, False, False, ABILITY_ID.BUILD_BUNKER.value, 0, [ UNIT_TYPEID.TERRAN_SCV.value ], [ UNIT_TYPEID.TERRAN_BARRACKS.value, UNIT_TYPEID.TERRAN_BARRACKSFLYING.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.TERRAN_ENGINEERINGBAY.value] =    TypeData( RACE.Terran, 125, 0, 0, 0, True, True, False, False, False, False, False, ABILITY_ID.BUILD_ENGINEERINGBAY.value, 0, [ UNIT_TYPEID.TERRAN_SCV.value ], [ UNIT_TYPEID.TERRAN_COMMANDCENTER.value, UNIT_TYPEID.TERRAN_COMMANDCENTERFLYING.value, UNIT_TYPEID.TERRAN_PLANETARYFORTRESS.value, UNIT_TYPEID.TERRAN_ORBITALCOMMAND.value, UNIT_TYPEID.TERRAN_ORBITALCOMMANDFLYING.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.TERRAN_MISSILETURRET.value] =     TypeData( RACE.Terran, 100, 0, 0, 0, True, True, False, False, False, False, False, ABILITY_ID.BUILD_MISSILETURRET.value, 0, [ UNIT_TYPEID.TERRAN_SCV.value ], [ UNIT_TYPEID.TERRAN_ENGINEERINGBAY.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.TERRAN_ORBITALCOMMAND.value] =    TypeData( RACE.Terran, 550, 0, 0, 0, True, True, False, False, False, False, False, ABILITY_ID.MORPH_ORBITALCOMMAND.value, 0, [ UNIT_TYPEID.TERRAN_COMMANDCENTER.value ], [ UNIT_TYPEID.TERRAN_BARRACKS.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.TERRAN_PLANETARYFORTRESS.value] = TypeData( RACE.Terran, 550, 150, 0, 0, True, True, False, False, False, False, False, ABILITY_ID.MORPH_PLANETARYFORTRESS.value, 0, [ UNIT_TYPEID.TERRAN_COMMANDCENTER.value ], [ UNIT_TYPEID.TERRAN_ENGINEERINGBAY.value ], [] )

    # Terran Addons                                                                             m  g  s  t  unit  bld   wrk    rfn    sup    hall   add
        self.m_unitTypeData[UNIT_TYPEID.TERRAN_BARRACKSREACTOR.value] =   TypeData( RACE.Terran, 50, 50, 0, 0, True, True, False, False, False, False, True, ABILITY_ID.BUILD_REACTOR_BARRACKS.value, 0, [ UNIT_TYPEID.TERRAN_BARRACKS.value ], [], [] )
        self.m_unitTypeData[UNIT_TYPEID.TERRAN_BARRACKSTECHLAB.value] =   TypeData( RACE.Terran, 50, 25, 0, 0, True, True, False, False, False, False, True, ABILITY_ID.BUILD_TECHLAB_BARRACKS.value, 0, [ UNIT_TYPEID.TERRAN_BARRACKS.value ], [], [] )
        self.m_unitTypeData[UNIT_TYPEID.TERRAN_FACTORYREACTOR.value] =    TypeData( RACE.Terran, 50, 50, 0, 0, True, True, False, False, False, False, True, ABILITY_ID.BUILD_REACTOR_FACTORY.value, 0, [ UNIT_TYPEID.TERRAN_FACTORY.value ], [], [] )
        self.m_unitTypeData[UNIT_TYPEID.TERRAN_FACTORYTECHLAB.value] =    TypeData( RACE.Terran, 50, 25, 0, 0, True, True, False, False, False, False, True, ABILITY_ID.BUILD_TECHLAB_FACTORY.value, 0, [ UNIT_TYPEID.TERRAN_FACTORY.value ], [], [] )
        self.m_unitTypeData[UNIT_TYPEID.TERRAN_STARPORTREACTOR.value] =   TypeData( RACE.Terran, 50, 50, 0, 0, True, True, False, False, False, False, True, ABILITY_ID.BUILD_REACTOR_STARPORT.value, 0, [ UNIT_TYPEID.TERRAN_STARPORT.value ], [], [] )
        self.m_unitTypeData[UNIT_TYPEID.TERRAN_STARPORTTECHLAB.value] =   TypeData( RACE.Terran, 50, 25, 0, 0, True, True, False, False, False, False, True, ABILITY_ID.BUILD_TECHLAB_STARPORT.value, 0, [ UNIT_TYPEID.TERRAN_STARPORT.value ], [], [] )

    # Terran Equivalent Buildings
        self.m_unitTypeData[UNIT_TYPEID.TERRAN_SUPPLYDEPOTLOWERED.value] =           self.m_unitTypeData[UNIT_TYPEID.TERRAN_SUPPLYDEPOT.value]
        self.m_unitTypeData[UNIT_TYPEID.TERRAN_BARRACKSFLYING.value] =               self.m_unitTypeData[UNIT_TYPEID.TERRAN_BARRACKS.value]
        self.m_unitTypeData[UNIT_TYPEID.TERRAN_COMMANDCENTERFLYING.value] =          self.m_unitTypeData[UNIT_TYPEID.TERRAN_COMMANDCENTER.value]
        self.m_unitTypeData[UNIT_TYPEID.TERRAN_FACTORYFLYING.value] =                self.m_unitTypeData[UNIT_TYPEID.TERRAN_FACTORY.value]
        self.m_unitTypeData[UNIT_TYPEID.TERRAN_ORBITALCOMMANDFLYING.value] =         self.m_unitTypeData[UNIT_TYPEID.TERRAN_ORBITALCOMMAND.value]
        self.m_unitTypeData[UNIT_TYPEID.TERRAN_STARPORTFLYING.value] =               self.m_unitTypeData[UNIT_TYPEID.TERRAN_STARPORT.value]

    # Terran Units                                                                              m  g  s  t  unit  bld    wrk    rfn    sup    hall   add
        self.m_unitTypeData[UNIT_TYPEID.TERRAN_SCV.value] =               TypeData( RACE.Terran, 50, 0, 1, 0, True, False,  True, False, False, False, False, ABILITY_ID.TRAIN_SCV.value, 0, [ UNIT_TYPEID.TERRAN_COMMANDCENTER.value ], [], [] )
        self.m_unitTypeData[UNIT_TYPEID.TERRAN_GHOST.value] =             TypeData( RACE.Terran, 200, 100, 2, 0, True, False, False, False, False, False, False, ABILITY_ID.TRAIN_GHOST.value, 0, [ UNIT_TYPEID.TERRAN_BARRACKS.value ], [], [] )
        self.m_unitTypeData[UNIT_TYPEID.TERRAN_MARAUDER.value] =          TypeData( RACE.Terran, 100, 25, 2, 0, True, False, False, False, False, False, False, ABILITY_ID.TRAIN_MARAUDER.value, 0, [ UNIT_TYPEID.TERRAN_BARRACKS.value ], [ UNIT_TYPEID.TERRAN_TECHLAB.value, UNIT_TYPEID.TERRAN_BARRACKSTECHLAB.value, UNIT_TYPEID.TERRAN_FACTORYTECHLAB.value, UNIT_TYPEID.TERRAN_STARPORTTECHLAB.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.TERRAN_MARINE.value] =            TypeData( RACE.Terran, 50, 0, 1, 0, True, False, False, False, False, False, False, ABILITY_ID.TRAIN_MARINE.value, 0, [ UNIT_TYPEID.TERRAN_BARRACKS.value ], [], [] )
        self.m_unitTypeData[UNIT_TYPEID.TERRAN_REAPER.value] =            TypeData( RACE.Terran, 50, 50, 1, 0, True, False, False, False, False, False, False, ABILITY_ID.TRAIN_REAPER.value, 0, [ UNIT_TYPEID.TERRAN_BARRACKS.value ], [], [] )
        self.m_unitTypeData[UNIT_TYPEID.TERRAN_HELLION.value] =           TypeData( RACE.Terran, 100, 0, 2, 0, True, False, False, False, False, False, False, ABILITY_ID.TRAIN_HELLION.value, 0, [ UNIT_TYPEID.TERRAN_FACTORY.value ], [], [] )
        self.m_unitTypeData[UNIT_TYPEID.TERRAN_CYCLONE.value] =           TypeData( RACE.Terran, 150, 100, 3, 0, True, False, False, False, False, False, False, ABILITY_ID.TRAIN_CYCLONE.value, 0, [ UNIT_TYPEID.TERRAN_FACTORY.value ], [], [] )
        self.m_unitTypeData[UNIT_TYPEID.TERRAN_SIEGETANK.value] =         TypeData( RACE.Terran, 150, 125, 3, 0, True, False, False, False, False, False, False, ABILITY_ID.TRAIN_SIEGETANK.value, 0, [ UNIT_TYPEID.TERRAN_FACTORY.value ], [ UNIT_TYPEID.TERRAN_TECHLAB.value, UNIT_TYPEID.TERRAN_BARRACKSTECHLAB.value, UNIT_TYPEID.TERRAN_FACTORYTECHLAB.value, UNIT_TYPEID.TERRAN_STARPORTTECHLAB.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.TERRAN_THOR.value] =              TypeData( RACE.Terran, 300, 200, 6, 0, True, False, False, False, False, False, False, ABILITY_ID.TRAIN_THOR.value, 0, [ UNIT_TYPEID.TERRAN_FACTORY.value ], [], []  )
        self.m_unitTypeData[UNIT_TYPEID.TERRAN_WIDOWMINE.value] =         TypeData( RACE.Terran, 75, 25, 2, 0, True, False, False, False, False, False, False, ABILITY_ID.TRAIN_WIDOWMINE.value, 0, [ UNIT_TYPEID.TERRAN_FACTORY.value ], [], [] )
        self.m_unitTypeData[UNIT_TYPEID.TERRAN_NUKE.value] =              TypeData( RACE.Terran, 100, 100, 0, 0, True, False, False, False, False, False, False, ABILITY_ID.BUILD_NUKE.value, 0, [ UNIT_TYPEID.TERRAN_GHOSTACADEMY.value ], [], [] )
        self.m_unitTypeData[UNIT_TYPEID.TERRAN_BANSHEE.value] =           TypeData( RACE.Terran, 150, 100, 3, 0, True, False, False, False, False, False, False, ABILITY_ID.TRAIN_BANSHEE.value, 0, [ UNIT_TYPEID.TERRAN_STARPORT.value ], [ UNIT_TYPEID.TERRAN_TECHLAB.value, UNIT_TYPEID.TERRAN_BARRACKSTECHLAB.value, UNIT_TYPEID.TERRAN_FACTORYTECHLAB.value, UNIT_TYPEID.TERRAN_STARPORTTECHLAB.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.TERRAN_BATTLECRUISER.value] =     TypeData( RACE.Terran, 400, 300, 6, 0, True, False, False, False, False, False, False, ABILITY_ID.TRAIN_BATTLECRUISER.value, 0, [ UNIT_TYPEID.TERRAN_STARPORT.value ], [], [] )
        self.m_unitTypeData[UNIT_TYPEID.TERRAN_LIBERATOR.value] =         TypeData( RACE.Terran, 150, 150, 3, 0, True, False, False, False, False, False, False, ABILITY_ID.TRAIN_LIBERATOR.value, 0, [ UNIT_TYPEID.TERRAN_STARPORT.value ], [], [] )
        self.m_unitTypeData[UNIT_TYPEID.TERRAN_VIKINGFIGHTER.value] =     TypeData( RACE.Terran, 150, 75, 2, 0, True, False, False, False, False, False, False, ABILITY_ID.TRAIN_VIKINGFIGHTER.value, 0, [ UNIT_TYPEID.TERRAN_STARPORT.value ], [], [] )
        self.m_unitTypeData[UNIT_TYPEID.TERRAN_RAVEN.value] =             TypeData( RACE.Terran, 100, 200, 2, 0, True, False, False, False, False, False, False, ABILITY_ID.TRAIN_RAVEN.value, 0, [ UNIT_TYPEID.TERRAN_STARPORT.value ], [ UNIT_TYPEID.TERRAN_TECHLAB.value, UNIT_TYPEID.TERRAN_BARRACKSTECHLAB.value, UNIT_TYPEID.TERRAN_FACTORYTECHLAB.value, UNIT_TYPEID.TERRAN_STARPORTTECHLAB.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.TERRAN_MEDIVAC.value] =           TypeData( RACE.Terran, 100, 100, 2, 0, True, False, False, False, False, False, False, ABILITY_ID.TRAIN_MEDIVAC.value, 0, [ UNIT_TYPEID.TERRAN_STARPORT.value ], [], [] )
        self.m_unitTypeData[UNIT_TYPEID.TERRAN_MULE.value] =              TypeData( RACE.Terran, 50, 0, 0, 0, True, False, False, False, False, False, False, ABILITY_ID.EFFECT_CALLDOWNMULE.value, 0, [ UNIT_TYPEID.TERRAN_ORBITALCOMMAND.value ], [], [] )

    # Zerg Buildings                                                                          m  g  s  t  unit  bld   wrk    rfn    sup    hall   add
        self.m_unitTypeData[UNIT_TYPEID.ZERG_HATCHERY.value] =            TypeData( RACE.Zerg, 300, 0, 0, self.fps*100, True, True, False, False, False,  True, False, ABILITY_ID.BUILD_HATCHERY.value, 0, [ UNIT_TYPEID.ZERG_DRONE.value ], [], [] )
        self.m_unitTypeData[UNIT_TYPEID.ZERG_EXTRACTOR.value] =           TypeData( RACE.Zerg, 25, 0, 0, self.fps*30, True, True, False,  True, False, False, False, ABILITY_ID.BUILD_EXTRACTOR.value, 0, [ UNIT_TYPEID.ZERG_DRONE.value ], [], [] )
        self.m_unitTypeData[UNIT_TYPEID.ZERG_SPAWNINGPOOL.value] =        TypeData( RACE.Zerg, 200, 0, 0, self.fps*65, True, True, False, False, False, False, False, ABILITY_ID.BUILD_SPAWNINGPOOL.value, 0, [ UNIT_TYPEID.ZERG_DRONE.value ], [ UNIT_TYPEID.ZERG_HATCHERY.value, UNIT_TYPEID.ZERG_LAIR.value, UNIT_TYPEID.ZERG_HIVE.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.ZERG_EVOLUTIONCHAMBER.value] =    TypeData( RACE.Zerg, 75, 0, 0, self.fps*35, True, True, False, False, False, False, False, ABILITY_ID.BUILD_EVOLUTIONCHAMBER.value, 0, [ UNIT_TYPEID.ZERG_DRONE.value ], [ UNIT_TYPEID.ZERG_HATCHERY.value, UNIT_TYPEID.ZERG_LAIR.value, UNIT_TYPEID.ZERG_HIVE.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.ZERG_BANELINGNEST.value] =        TypeData( RACE.Zerg, 100, 50, 0, self.fps*60, True, True, False, False, False, False, False, ABILITY_ID.BUILD_BANELINGNEST.value, 0, [ UNIT_TYPEID.ZERG_DRONE.value ], [ UNIT_TYPEID.ZERG_SPAWNINGPOOL.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.ZERG_HYDRALISKDEN.value] =        TypeData( RACE.Zerg, 100, 100, 0, self.fps*40, True, True, False, False, False, False, False, ABILITY_ID.BUILD_HYDRALISKDEN.value, 0, [ UNIT_TYPEID.ZERG_DRONE.value ], [ UNIT_TYPEID.ZERG_LAIR.value, UNIT_TYPEID.ZERG_HIVE.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.ZERG_INFESTATIONPIT.value] =      TypeData( RACE.Zerg, 100, 100, 0, self.fps*50, True, True, False, False, False, False, False, ABILITY_ID.BUILD_INFESTATIONPIT.value, 0, [ UNIT_TYPEID.ZERG_DRONE.value ], [ UNIT_TYPEID.ZERG_LAIR.value, UNIT_TYPEID.ZERG_HIVE.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.ZERG_NYDUSCANAL.value] =          TypeData( RACE.Zerg, 100, 100, 0, self.fps*20, True, True, False, False, False, False, False, ABILITY_ID.BUILD_NYDUSWORM.value, 0, [ UNIT_TYPEID.ZERG_NYDUSNETWORK.value ], [ UNIT_TYPEID.ZERG_NYDUSNETWORK.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.ZERG_NYDUSNETWORK.value] =        TypeData( RACE.Zerg, 150, 200, 0, self.fps*50, True, True, False, False, False, False, False, ABILITY_ID.BUILD_NYDUSNETWORK.value, 0, [ UNIT_TYPEID.ZERG_DRONE.value ], [ UNIT_TYPEID.ZERG_LAIR.value, UNIT_TYPEID.ZERG_HIVE.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.ZERG_ROACHWARREN.value] =         TypeData( RACE.Zerg, 150, 0, 0, self.fps*55, True, True, False, False, False, False, False, ABILITY_ID.BUILD_ROACHWARREN.value, 0, [ UNIT_TYPEID.ZERG_DRONE.value ], [ UNIT_TYPEID.ZERG_SPAWNINGPOOL.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.ZERG_SPINECRAWLER.value] =        TypeData( RACE.Zerg, 100, 0, 0, self.fps*50, True, True, False, False, False, False, False, ABILITY_ID.BUILD_SPINECRAWLER.value, 0, [ UNIT_TYPEID.ZERG_DRONE.value ], [ UNIT_TYPEID.ZERG_SPAWNINGPOOL.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.ZERG_SPIRE.value] =               TypeData( RACE.Zerg, 200, 200, 0, self.fps*100, True, True, False, False, False, False, False, ABILITY_ID.BUILD_SPIRE.value, 0, [ UNIT_TYPEID.ZERG_DRONE.value ], [ UNIT_TYPEID.ZERG_LAIR.value, UNIT_TYPEID.ZERG_HIVE.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.ZERG_GREATERSPIRE.value] =        TypeData( RACE.Zerg, 100, 150, 0, self.fps*100, True, True, False, False, False, False, False, ABILITY_ID.MORPH_GREATERSPIRE.value, 0, [ UNIT_TYPEID.ZERG_SPIRE.value ], [ UNIT_TYPEID.ZERG_HIVE.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.ZERG_SPORECRAWLER.value] =        TypeData( RACE.Zerg, 75, 0, 0, self.fps*30, True, True, False, False, False, False, False, ABILITY_ID.BUILD_SPORECRAWLER.value, 0, [ UNIT_TYPEID.ZERG_DRONE.value ], [ UNIT_TYPEID.ZERG_SPAWNINGPOOL.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.ZERG_ULTRALISKCAVERN.value] =     TypeData( RACE.Zerg, 150, 200, 0, self.fps*65, True, True, False, False, False, False, False, ABILITY_ID.BUILD_ULTRALISKCAVERN.value, 0, [ UNIT_TYPEID.ZERG_DRONE.value ], [ UNIT_TYPEID.ZERG_HIVE.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.ZERG_LAIR.value] =                TypeData( RACE.Zerg, 150, 100, 0, self.fps*80, True, True, False, False, False,  True, False, ABILITY_ID.MORPH_LAIR.value, 0, [ UNIT_TYPEID.ZERG_HATCHERY.value ], [UNIT_TYPEID.ZERG_SPAWNINGPOOL.value], [] )
        self.m_unitTypeData[UNIT_TYPEID.ZERG_HIVE.value] =                TypeData( RACE.Zerg, 200, 150, 0, self.fps*100, True, True, False, False, False,  True, False, ABILITY_ID.MORPH_HIVE.value, 0, [ UNIT_TYPEID.ZERG_LAIR.value ], [ UNIT_TYPEID.ZERG_INFESTATIONPIT.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.ZERG_LURKERDENMP.value] =         TypeData( RACE.Zerg, 150, 150, 0, self.fps*120, True, True, False, False, False, False, False, ABILITY_ID.MORPH_LURKERDEN.value, 0,  [UNIT_TYPEID.ZERG_HYDRALISKDEN.value], [], [] )

    # Zerg Units                                                                              m  g  s  t  unit  bld    wrk    rfn    sup    hall   add
        self.m_unitTypeData[UNIT_TYPEID.ZERG_OVERLORD.value] =            TypeData( RACE.Zerg, 100, 0, 0, self.fps*25, True, False, False, False,  True, False, False, ABILITY_ID.TRAIN_OVERLORD.value, 0, [ UNIT_TYPEID.ZERG_LARVA.value ], [], [] )
        self.m_unitTypeData[UNIT_TYPEID.ZERG_BANELING.value] =            TypeData( RACE.Zerg, 25, 25, 0, self.fps*20, True, False, False, False, False, False, False, ABILITY_ID.TRAIN_BANELING.value, 0, [ UNIT_TYPEID.ZERG_ZERGLING.value ], [ UNIT_TYPEID.ZERG_BANELINGNEST.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.ZERG_CORRUPTOR.value] =           TypeData( RACE.Zerg, 150, 100, 2, self.fps*40, True, False, False, False, False, False, False, ABILITY_ID.TRAIN_CORRUPTOR.value, 0, [ UNIT_TYPEID.ZERG_LARVA.value ], [ UNIT_TYPEID.ZERG_SPIRE.value, UNIT_TYPEID.ZERG_GREATERSPIRE.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.ZERG_DRONE.value] =               TypeData( RACE.Zerg, 50, 0, 1, self.fps*17, True, False, False, False, False, False, False, ABILITY_ID.TRAIN_DRONE.value, 0, [ UNIT_TYPEID.ZERG_LARVA.value ], [], [] )
        self.m_unitTypeData[UNIT_TYPEID.ZERG_HYDRALISK.value] =           TypeData( RACE.Zerg, 100, 50, 2, self.fps*33, True, False, False, False, False, False, False, ABILITY_ID.TRAIN_HYDRALISK.value, 0, [ UNIT_TYPEID.ZERG_LARVA.value ], [ UNIT_TYPEID.ZERG_HYDRALISKDEN.value, UNIT_TYPEID.ZERG_LURKERDENMP.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.ZERG_INFESTOR.value] =            TypeData( RACE.Zerg, 100, 150, 2, self.fps*50, True, False, False, False, False, False, False, ABILITY_ID.TRAIN_INFESTOR.value, 0, [ UNIT_TYPEID.ZERG_LARVA.value ], [ UNIT_TYPEID.ZERG_INFESTATIONPIT.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.ZERG_MUTALISK.value] =            TypeData( RACE.Zerg, 100, 100, 2, self.fps*33, True, False, False, False, False, False, False, ABILITY_ID.TRAIN_MUTALISK.value, 0, [ UNIT_TYPEID.ZERG_LARVA.value ], [ UNIT_TYPEID.ZERG_SPIRE.value, UNIT_TYPEID.ZERG_GREATERSPIRE.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.ZERG_ROACH.value] =               TypeData( RACE.Zerg, 75, 25, 2, self.fps*27, True, False, False, False, False, False, False, ABILITY_ID.TRAIN_ROACH.value, 0, [ UNIT_TYPEID.ZERG_LARVA.value ], [ UNIT_TYPEID.ZERG_ROACHWARREN.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.ZERG_SWARMHOSTMP.value] =         TypeData( RACE.Zerg, 100, 75, 3, self.fps*40, True, False, False, False, False, False, False, ABILITY_ID.TRAIN_SWARMHOST.value, 0, [ UNIT_TYPEID.ZERG_LARVA.value ], [ UNIT_TYPEID.ZERG_INFESTATIONPIT.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.ZERG_ULTRALISK.value] =           TypeData( RACE.Zerg, 300, 200, 6, self.fps*55, True, False, False, False, False, False, False, ABILITY_ID.TRAIN_ULTRALISK.value, 0, [ UNIT_TYPEID.ZERG_LARVA.value ], [ UNIT_TYPEID.ZERG_ULTRALISKCAVERN.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.ZERG_VIPER.value] =               TypeData( RACE.Zerg, 100, 200, 3, self.fps*40, True, False, False, False, False, False, False, ABILITY_ID.TRAIN_VIPER.value, 0, [ UNIT_TYPEID.ZERG_LARVA.value ], [ UNIT_TYPEID.ZERG_HIVE.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.ZERG_ZERGLING.value] =            TypeData( RACE.Zerg, 50, 0, 1, self.fps*24, True, False, False, False, False, False, False, ABILITY_ID.TRAIN_ZERGLING.value, 0, [ UNIT_TYPEID.ZERG_LARVA.value ], [ UNIT_TYPEID.ZERG_SPAWNINGPOOL.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.ZERG_QUEEN.value] =               TypeData( RACE.Zerg, 150, 0, 2, self.fps*50, True, False, False, False, False, False, False, ABILITY_ID.TRAIN_QUEEN.value, 0, [ UNIT_TYPEID.ZERG_HATCHERY.value, UNIT_TYPEID.ZERG_LAIR.value, UNIT_TYPEID.ZERG_HIVE.value ], [ UNIT_TYPEID.ZERG_SPAWNINGPOOL.value ], [] )
        self.m_unitTypeData[UNIT_TYPEID.ZERG_LARVA.value] =               TypeData( RACE.Zerg, 0, 0, 0, self.fps*15, True, False, False, False, False, False, False, 0, 0, [], [], [] )
        self.m_unitTypeData[UNIT_TYPEID.ZERG_EGG.value] =                 TypeData( RACE.Zerg, 0, 0, 0, 0, True, False, False, False, False, False, False, 0, 0, [], [], [] )
        self.m_unitTypeData[UNIT_TYPEID.ZERG_LURKERMP.value] =            TypeData( RACE.Zerg, 50, 100, 1, self.fps*25, True, True, False, False, False, False, False, ABILITY_ID.MORPH_LURKER.value, 0,  [UNIT_TYPEID.ZERG_HYDRALISK.value], [UNIT_TYPEID.ZERG_LURKERDENMP.value], [] )
        self.m_unitTypeData[UNIT_TYPEID.ZERG_RAVAGER.value] =             TypeData( RACE.Zerg, 25, 75, 1, self.fps*12, True, False, False, False, False, False, False, ABILITY_ID.MORPH_RAVAGER.value, 0,  [UNIT_TYPEID.ZERG_ROACH.value], [], [] )
        self.m_unitTypeData[UNIT_TYPEID.ZERG_BROODLORD.value] =           TypeData( RACE.Zerg, 150, 150, 2, self.fps*34, True, False, False, False, False, False, False, ABILITY_ID.MORPH_BROODLORD.value, 0,  [UNIT_TYPEID.ZERG_CORRUPTOR.value], [UNIT_TYPEID.ZERG_GREATERSPIRE.value], [] )
        self.m_unitTypeData[UNIT_TYPEID.ZERG_OVERSEER.value] =            TypeData( RACE.Zerg, 50, 50, 0, self.fps*17, True, False, False, False, True, False, False, ABILITY_ID.MORPH_OVERSEER.value, 0,  [UNIT_TYPEID.ZERG_OVERLORD.value], [ UNIT_TYPEID.ZERG_LAIR.value ], [] )

    # Set the Mineral / Gas cost of each unit from game core data
        data_raw = data_raw_3_16
        for unit_type in self.m_unitTypeData:
            mineralCost = data_raw.units[unit_type].mineral_cost
            gasCost = data_raw.units[unit_type].vespene_cost
            if self.m_unitTypeData[unit_type].race == RACE.Zerg:
                if (unit_type not in [UNIT_TYPEID.ZERG_LARVA.value,
                                      UNIT_TYPEID.ZERG_EGG.value,
                                      UNIT_TYPEID.ZERG_QUEEN.value,
                                      UNIT_TYPEID.ZERG_NYDUSCANAL.value,
                                      UNIT_TYPEID.ZERG_ZERGLING.value]):
                    builder_type = self.m_unitTypeData[unit_type].whatBuilds[0]
                    mineralCost -= data_raw.units[builder_type].mineral_cost
                    gasCost -= data_raw.units[builder_type].vespene_cost
                if (unit_type == UNIT_TYPEID.ZERG_ZERGLING.value):
                    mineralCost *= 2
                    gasCost *= 2
                #if (mineralCost != self.m_unitTypeData[unit_type].mineralCost or
                #            gasCost != self.m_unitTypeData[unit_type].gasCost):
                #    print('Data for unit {} inconsistent!'.format(unit_type))
                self.m_unitTypeData[unit_type].mineralCost = mineralCost
                self.m_unitTypeData[unit_type].gasCost = gasCost

    # fix the cumulative prices of morphed buildings
    #    self.m_unitTypeData[UNIT_TYPEID.ZERG_HIVE.value].mineralCost -= self.getUnitData(UNIT_TYPEID.ZERG_LAIR.value).mineralCost
    #    self.m_unitTypeData[UNIT_TYPEID.ZERG_LAIR.value].mineralCost -= self.getUnitData(UNIT_TYPEID.ZERG_HATCHERY.value).mineralCost
        self.m_unitTypeData[UNIT_TYPEID.TERRAN_PLANETARYFORTRESS.value].mineralCost -= self.getUnitData(UNIT_TYPEID.TERRAN_COMMANDCENTER.value).mineralCost
        self.m_unitTypeData[UNIT_TYPEID.TERRAN_ORBITALCOMMAND.value].mineralCost -= self.getUnitData(UNIT_TYPEID.TERRAN_COMMANDCENTER.value).mineralCost
    #    self.m_unitTypeData[UNIT_TYPEID.ZERG_GREATERSPIRE.value].mineralCost -= self.getUnitData(UNIT_TYPEID.ZERG_SPIRE.value).mineralCost

    def initUpgradeData(self):
    # 0 data for null / error return

    # Terran Upgrades
        self.m_upgradeData[UPGRADE_ID.BANSHEECLOAK.value] =               TypeData( RACE.Terran, 100, 100, 0, self.fps*110, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_BANSHEECLOAKINGFIELD.value, 0, [ UNIT_TYPEID.TERRAN_SCV.value ], [], [] )
        self.m_upgradeData[UPGRADE_ID.BANSHEESPEED.value] =               TypeData( RACE.Terran, 200, 200, 0, self.fps*170, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_BANSHEEHYPERFLIGHTROTORS.value, 0, [ UNIT_TYPEID.TERRAN_SCV.value ], [], [] )
        self.m_upgradeData[UPGRADE_ID.BATTLECRUISERENABLESPECIALIZATIONS.value] = TypeData( RACE.Terran, 150, 150, 0,  self.fps*60, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_BATTLECRUISERWEAPONREFIT.value, 0, [ UNIT_TYPEID.TERRAN_SCV.value ], [], [] )
        self.m_upgradeData[UPGRADE_ID.DRILLCLAWS.value] =                 TypeData( RACE.Terran, 150, 150, 0, self.fps*110, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_DRILLINGCLAWS.value, 0, [ UNIT_TYPEID.TERRAN_SCV.value ], [], [] )
        self.m_upgradeData[UPGRADE_ID.HIGHCAPACITYBARRELS.value] =        TypeData( RACE.Terran, 150, 150, 0, self.fps*110, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_INFERNALPREIGNITER.value, 0, [ UNIT_TYPEID.TERRAN_SCV.value ], [], [] )
        self.m_upgradeData[UPGRADE_ID.HISECAUTOTRACKING.value] =          TypeData( RACE.Terran, 100, 100, 0, self.fps*80, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_HISECAUTOTRACKING.value, 0, [ UNIT_TYPEID.TERRAN_ENGINEERINGBAY.value ], [], [] )
        self.m_upgradeData[UPGRADE_ID.LIBERATORAGRANGEUPGRADE.value] =    TypeData( RACE.Terran, 150, 150, 0, self.fps*110, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_ADVANCEDBALLISTICS.value, 0, [ UNIT_TYPEID.TERRAN_STARPORTTECHLAB.value ], [], [] )
        self.m_upgradeData[UPGRADE_ID.MAGFIELDLAUNCHERS.value] =          TypeData( RACE.Terran, 100, 100, 0, self.fps*110, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_MAGFIELDLAUNCHERS.value, 0, [ UNIT_TYPEID.TERRAN_SCV.value ], [], [] )
        self.m_upgradeData[UPGRADE_ID.MEDIVACINCREASESPEEDBOOST.value] =  TypeData( RACE.Terran, 100, 100, 0, self.fps*80, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_HIGHCAPACITYFUELTANKS.value, 0, [ UNIT_TYPEID.TERRAN_SCV.value ], [], [] )
        self.m_upgradeData[UPGRADE_ID.NEOSTEELFRAME.value] =              TypeData( RACE.Terran, 100, 100, 0, self.fps*110, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_NEOSTEELFRAME.value, 0, [ UNIT_TYPEID.TERRAN_ENGINEERINGBAY.value ], [], [] )
        self.m_upgradeData[UPGRADE_ID.PERSONALCLOAKING.value] =           TypeData( RACE.Terran, 150, 150, 0, self.fps*120, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_PERSONALCLOAKING.value, 0, [ UNIT_TYPEID.TERRAN_GHOSTACADEMY.value ], [], [] )
        self.m_upgradeData[UPGRADE_ID.PUNISHERGRENADES.value] =           TypeData( RACE.Terran,  50,  50, 0,  self.fps*60, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_CONCUSSIVESHELLS.value, 0, [ UNIT_TYPEID.TERRAN_SCV.value ], [], [] )
        self.m_upgradeData[UPGRADE_ID.RAVENCORVIDREACTOR.value] =         TypeData( RACE.Terran, 150, 150, 0, self.fps*110, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_RAVENCORVIDREACTOR.value, 0, [ UNIT_TYPEID.TERRAN_SCV.value ], [], [] )
        self.m_upgradeData[UPGRADE_ID.RAVENRECALIBRATEDEXPLOSIVES.value] = TypeData( RACE.Terran, 150, 150, 0, self.fps*110, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_RAVENRECALIBRATEDEXPLOSIVES.value, 0, [ UNIT_TYPEID.TERRAN_SCV.value ], [], [] )
        self.m_upgradeData[UPGRADE_ID.SHIELDWALL.value] =                 TypeData( RACE.Terran, 100, 100, 0, self.fps*110, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_COMBATSHIELD.value, 0, [ UNIT_TYPEID.TERRAN_BARRACKSTECHLAB.value ], [], [] )
        self.m_upgradeData[UPGRADE_ID.STIMPACK.value] =                   TypeData( RACE.Terran, 100, 100, 0, self.fps*170, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_STIMPACK.value, 0, [ UNIT_TYPEID.TERRAN_SCV.value ], [], [] )
        self.m_upgradeData[UPGRADE_ID.TERRANBUILDINGARMOR.value] =        TypeData( RACE.Terran, 150, 150, 0, self.fps*140, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_TERRANSTRUCTUREARMORUPGRADE.value, 0, [ UNIT_TYPEID.TERRAN_ENGINEERINGBAY.value ], [], [] )
        self.m_upgradeData[UPGRADE_ID.TERRANINFANTRYARMORSLEVEL1.value] = TypeData( RACE.Terran, 100, 100, 0, self.fps*160, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_TERRANINFANTRYARMORLEVEL1.value, 0, [ UNIT_TYPEID.TERRAN_ENGINEERINGBAY.value ], [], [] )
        self.m_upgradeData[UPGRADE_ID.TERRANINFANTRYARMORSLEVEL2.value] = TypeData( RACE.Terran, 175, 175, 0, self.fps*190, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_TERRANINFANTRYARMORLEVEL2.value, 0, [ UNIT_TYPEID.TERRAN_ENGINEERINGBAY.value ], [ UNIT_TYPEID.TERRAN_ARMORY.value ], [UPGRADE_ID.TERRANINFANTRYARMORSLEVEL1.value] )
        self.m_upgradeData[UPGRADE_ID.TERRANINFANTRYARMORSLEVEL3.value] = TypeData( RACE.Terran, 250, 250, 0, self.fps*220, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_TERRANINFANTRYARMORLEVEL3.value, 0, [ UNIT_TYPEID.TERRAN_ENGINEERINGBAY.value ], [ UNIT_TYPEID.TERRAN_ARMORY.value ], [UPGRADE_ID.TERRANINFANTRYARMORSLEVEL2.value] )
        self.m_upgradeData[UPGRADE_ID.TERRANINFANTRYWEAPONSLEVEL1.value] = TypeData( RACE.Terran, 100, 100, 0, self.fps*160, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_TERRANINFANTRYWEAPONSLEVEL1.value, 0, [ UNIT_TYPEID.TERRAN_ENGINEERINGBAY.value ], [], [] )
        self.m_upgradeData[UPGRADE_ID.TERRANINFANTRYWEAPONSLEVEL2.value] = TypeData( RACE.Terran, 175, 175, 0, self.fps*190, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_TERRANINFANTRYWEAPONSLEVEL2.value, 0, [ UNIT_TYPEID.TERRAN_ENGINEERINGBAY.value ], [ UNIT_TYPEID.TERRAN_ARMORY.value ], [UPGRADE_ID.TERRANINFANTRYWEAPONSLEVEL1.value] )
        self.m_upgradeData[UPGRADE_ID.TERRANINFANTRYWEAPONSLEVEL3.value] = TypeData( RACE.Terran, 250, 250, 0, self.fps*220, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_TERRANINFANTRYWEAPONSLEVEL3.value, 0, [ UNIT_TYPEID.TERRAN_ENGINEERINGBAY.value ], [ UNIT_TYPEID.TERRAN_ARMORY.value ], [UPGRADE_ID.TERRANINFANTRYWEAPONSLEVEL2.value] )
        self.m_upgradeData[UPGRADE_ID.TERRANSHIPWEAPONSLEVEL1.value] =    TypeData( RACE.Terran, 100, 100, 0, self.fps*160, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_TERRANSHIPWEAPONSLEVEL1.value, 0, [ UNIT_TYPEID.TERRAN_ARMORY.value ], [], [] )
        self.m_upgradeData[UPGRADE_ID.TERRANSHIPWEAPONSLEVEL2.value] =    TypeData( RACE.Terran, 175, 175, 0, self.fps*190, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_TERRANSHIPWEAPONSLEVEL2.value, 0, [ UNIT_TYPEID.TERRAN_ARMORY.value ], [], [UPGRADE_ID.TERRANSHIPWEAPONSLEVEL1.value] )
        self.m_upgradeData[UPGRADE_ID.TERRANSHIPWEAPONSLEVEL3.value] =    TypeData( RACE.Terran, 250, 250, 0, self.fps*220, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_TERRANSHIPWEAPONSLEVEL3.value, 0, [ UNIT_TYPEID.TERRAN_ARMORY.value ], [], [UPGRADE_ID.TERRANSHIPWEAPONSLEVEL2.value] )
        self.m_upgradeData[UPGRADE_ID.TERRANVEHICLEANDSHIPARMORSLEVEL1.value] =  TypeData( RACE.Terran, 100, 100, 0, self.fps*160, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_TERRANVEHICLEANDSHIPPLATINGLEVEL1.value, 0, [ UNIT_TYPEID.TERRAN_ARMORY.value ], [], [] )
        self.m_upgradeData[UPGRADE_ID.TERRANVEHICLEANDSHIPARMORSLEVEL2.value] =  TypeData( RACE.Terran, 175, 175, 0, self.fps*190, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_TERRANVEHICLEANDSHIPPLATINGLEVEL2.value, 0, [ UNIT_TYPEID.TERRAN_ARMORY.value ], [], [UPGRADE_ID.TERRANVEHICLEANDSHIPARMORSLEVEL1.value] )
        self.m_upgradeData[UPGRADE_ID.TERRANVEHICLEANDSHIPARMORSLEVEL3.value] =  TypeData( RACE.Terran, 250, 250, 0, self.fps*220, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_TERRANVEHICLEANDSHIPPLATINGLEVEL3.value, 0, [ UNIT_TYPEID.TERRAN_ARMORY.value ], [], [UPGRADE_ID.TERRANVEHICLEANDSHIPARMORSLEVEL2.value] )
        self.m_upgradeData[UPGRADE_ID.TERRANVEHICLEWEAPONSLEVEL1.value] = TypeData( RACE.Terran, 100, 100, 0, self.fps*160, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_TERRANVEHICLEWEAPONSLEVEL1.value, 0, [ UNIT_TYPEID.TERRAN_ARMORY.value ], [], [] )
        self.m_upgradeData[UPGRADE_ID.TERRANVEHICLEWEAPONSLEVEL2.value] = TypeData( RACE.Terran, 175, 175, 0, self.fps*190, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_TERRANVEHICLEWEAPONSLEVEL2.value, 0, [ UNIT_TYPEID.TERRAN_ARMORY.value ], [], [UPGRADE_ID.TERRANVEHICLEWEAPONSLEVEL1.value] )
        self.m_upgradeData[UPGRADE_ID.TERRANVEHICLEWEAPONSLEVEL3.value] = TypeData( RACE.Terran, 250, 250, 0, self.fps*220, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_TERRANVEHICLEWEAPONSLEVEL3.value, 0, [ UNIT_TYPEID.TERRAN_ARMORY.value ], [], [UPGRADE_ID.TERRANVEHICLEWEAPONSLEVEL2.value] )

    # Protoss Upgrades
        self.m_upgradeData[UPGRADE_ID.ADEPTPIERCINGATTACK.value] =        TypeData( RACE.Protoss, 100, 100, 0, self.fps*140, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_ADEPTRESONATINGGLAIVES.value, 0, [ UNIT_TYPEID.PROTOSS_TWILIGHTCOUNCIL.value ], [], [] )
        self.m_upgradeData[UPGRADE_ID.BLINKTECH.value] =                  TypeData( RACE.Protoss, 150, 150, 0, self.fps*170, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_BLINK.value, 0,            [ UNIT_TYPEID.PROTOSS_TWILIGHTCOUNCIL.value ], [], [] )
        self.m_upgradeData[UPGRADE_ID.CARRIERLAUNCHSPEEDUPGRADE.value] =  TypeData( RACE.Protoss, 150, 150, 0, self.fps*80, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_INTERCEPTORGRAVITONCATAPULT.value, 0, [ UNIT_TYPEID.PROTOSS_FLEETBEACON.value ], [], [] )
        self.m_upgradeData[UPGRADE_ID.CHARGE.value] =                     TypeData( RACE.Protoss, 100, 100, 0, self.fps*140, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_CHARGE.value, 0,           [ UNIT_TYPEID.PROTOSS_TWILIGHTCOUNCIL.value ], [], [] )
        self.m_upgradeData[UPGRADE_ID.DARKTEMPLARBLINKUPGRADE.value] =    TypeData( RACE.Protoss, 100, 100, 0, self.fps*170, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_SHADOWSTRIKE.value, 0,     [ UNIT_TYPEID.PROTOSS_DARKSHRINE.value ], [], [] )
        self.m_upgradeData[UPGRADE_ID.EXTENDEDTHERMALLANCE.value] =       TypeData( RACE.Protoss, 200, 200, 0, self.fps*140, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_EXTENDEDTHERMALLANCE.value, 0,    [ UNIT_TYPEID.PROTOSS_ROBOTICSBAY.value ], [], [] )
        self.m_upgradeData[UPGRADE_ID.GRAVITICDRIVE.value] =              TypeData( RACE.Protoss, 100, 100, 0, self.fps*80, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_GRAVITICDRIVE.value, 0,    [ UNIT_TYPEID.PROTOSS_ROBOTICSBAY.value ], [], [] )
        self.m_upgradeData[UPGRADE_ID.OBSERVERGRAVITICBOOSTER.value] =    TypeData( RACE.Protoss, 100, 100, 0, self.fps*80, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_GRAVITICBOOSTER.value, 0,  [ UNIT_TYPEID.PROTOSS_ROBOTICSBAY.value ], [], [] )
        self.m_upgradeData[UPGRADE_ID.PHOENIXRANGEUPGRADE.value] =        TypeData( RACE.Protoss, 150, 150, 0, 1440, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_PHOENIXANIONPULSECRYSTALS.value, 0, [ UNIT_TYPEID.PROTOSS_FLEETBEACON.value ], [], [] )
        self.m_upgradeData[UPGRADE_ID.PROTOSSAIRARMORSLEVEL1.value] =     TypeData( RACE.Protoss, 150, 150, 0, self.fps*160, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_PROTOSSAIRARMORLEVEL1.value, 0,   [ UNIT_TYPEID.PROTOSS_CYBERNETICSCORE.value ], [], [] )
        self.m_upgradeData[UPGRADE_ID.PROTOSSAIRARMORSLEVEL2.value] =     TypeData( RACE.Protoss, 225, 225, 0, self.fps*190, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_PROTOSSAIRARMORLEVEL2.value, 0,   [ UNIT_TYPEID.PROTOSS_CYBERNETICSCORE.value ], [ UNIT_TYPEID.PROTOSS_FLEETBEACON.value ], [UPGRADE_ID.PROTOSSAIRARMORSLEVEL1.value] )
        self.m_upgradeData[UPGRADE_ID.PROTOSSAIRARMORSLEVEL3.value] =     TypeData( RACE.Protoss, 300, 300, 0, self.fps*220, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_PROTOSSAIRARMORLEVEL3.value, 0,   [ UNIT_TYPEID.PROTOSS_CYBERNETICSCORE.value ], [ UNIT_TYPEID.PROTOSS_FLEETBEACON.value ], [UPGRADE_ID.PROTOSSAIRARMORSLEVEL2.value] )
        self.m_upgradeData[UPGRADE_ID.PROTOSSAIRWEAPONSLEVEL1.value] =    TypeData( RACE.Protoss, 100, 100, 0, self.fps*160, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_PROTOSSAIRWEAPONSLEVEL1.value, 0, [ UNIT_TYPEID.PROTOSS_CYBERNETICSCORE.value ], [], [] )
        self.m_upgradeData[UPGRADE_ID.PROTOSSAIRWEAPONSLEVEL2.value] =    TypeData( RACE.Protoss, 175, 175, 0, self.fps*190, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_PROTOSSAIRWEAPONSLEVEL2.value, 0, [ UNIT_TYPEID.PROTOSS_CYBERNETICSCORE.value ], [ UNIT_TYPEID.PROTOSS_FLEETBEACON.value ], [UPGRADE_ID.PROTOSSAIRWEAPONSLEVEL1.value] )
        self.m_upgradeData[UPGRADE_ID.PROTOSSAIRWEAPONSLEVEL3.value] =    TypeData( RACE.Protoss, 250, 250, 0, self.fps*220, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_PROTOSSAIRWEAPONSLEVEL3.value, 0, [ UNIT_TYPEID.PROTOSS_CYBERNETICSCORE.value ], [ UNIT_TYPEID.PROTOSS_FLEETBEACON.value ], [UPGRADE_ID.PROTOSSAIRWEAPONSLEVEL3] )
        self.m_upgradeData[UPGRADE_ID.PROTOSSGROUNDARMORSLEVEL1.value] =  TypeData( RACE.Protoss, 100, 100, 0, self.fps*160, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_PROTOSSGROUNDARMORLEVEL1.value, 0, [ UNIT_TYPEID.PROTOSS_FORGE.value ], [], [] )
        self.m_upgradeData[UPGRADE_ID.PROTOSSGROUNDARMORSLEVEL2.value] =  TypeData( RACE.Protoss, 150, 150, 0, self.fps*190, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_PROTOSSGROUNDARMORLEVEL2.value, 0, [ UNIT_TYPEID.PROTOSS_FORGE.value ], [ UNIT_TYPEID.PROTOSS_TWILIGHTCOUNCIL.value ], [ UPGRADE_ID.PROTOSSGROUNDARMORSLEVEL1.value] )
        self.m_upgradeData[UPGRADE_ID.PROTOSSGROUNDARMORSLEVEL3.value] =  TypeData( RACE.Protoss, 200, 200, 0, self.fps*220, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_PROTOSSGROUNDARMORLEVEL3.value, 0, [ UNIT_TYPEID.PROTOSS_FORGE.value ], [ UNIT_TYPEID.PROTOSS_TWILIGHTCOUNCIL.value ], [ UPGRADE_ID.PROTOSSGROUNDARMORSLEVEL2.value] )
        self.m_upgradeData[UPGRADE_ID.PROTOSSGROUNDWEAPONSLEVEL1.value] = TypeData( RACE.Protoss, 100, 100, 0, self.fps*160, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_PROTOSSGROUNDWEAPONSLEVEL1.value, 0, [ UNIT_TYPEID.PROTOSS_FORGE.value ], [], [] )
        self.m_upgradeData[UPGRADE_ID.PROTOSSGROUNDWEAPONSLEVEL2.value] = TypeData( RACE.Protoss, 150, 150, 0, self.fps*190, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_PROTOSSGROUNDWEAPONSLEVEL2.value, 0, [ UNIT_TYPEID.PROTOSS_FORGE.value ], [ UNIT_TYPEID.PROTOSS_TWILIGHTCOUNCIL.value ], [ UPGRADE_ID.PROTOSSGROUNDWEAPONSLEVEL1.value] )
        self.m_upgradeData[UPGRADE_ID.PROTOSSGROUNDWEAPONSLEVEL3.value] = TypeData( RACE.Protoss, 200, 200, 0, self.fps*220, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_PROTOSSGROUNDWEAPONSLEVEL3.value, 0, [ UNIT_TYPEID.PROTOSS_FORGE.value ], [ UNIT_TYPEID.PROTOSS_TWILIGHTCOUNCIL.value ], [ UPGRADE_ID.PROTOSSGROUNDWEAPONSLEVEL2.value] )
        self.m_upgradeData[UPGRADE_ID.PROTOSSSHIELDSLEVEL1.value] =       TypeData( RACE.Protoss, 150, 150, 0, self.fps*160, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_PROTOSSSHIELDSLEVEL1.value, 0, [ UNIT_TYPEID.PROTOSS_FORGE.value ], [], [] )
        self.m_upgradeData[UPGRADE_ID.PROTOSSSHIELDSLEVEL2.value] =       TypeData( RACE.Protoss, 225, 225, 0, self.fps*190, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_PROTOSSSHIELDSLEVEL2.value, 0, [ UNIT_TYPEID.PROTOSS_FORGE.value ], [ UNIT_TYPEID.PROTOSS_TWILIGHTCOUNCIL.value ], [ UPGRADE_ID.PROTOSSSHIELDSLEVEL1.value] )
        self.m_upgradeData[UPGRADE_ID.PROTOSSSHIELDSLEVEL3.value] =       TypeData( RACE.Protoss, 300, 300, 0, self.fps*220, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_PROTOSSSHIELDSLEVEL3.value, 0, [ UNIT_TYPEID.PROTOSS_FORGE.value ], [ UNIT_TYPEID.PROTOSS_TWILIGHTCOUNCIL.value ], [ UPGRADE_ID.PROTOSSSHIELDSLEVEL2.value] )
        self.m_upgradeData[UPGRADE_ID.PSISTORMTECH.value] =               TypeData( RACE.Protoss, 200, 200, 0, self.fps*110, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_PSISTORM.value, 0, [ UNIT_TYPEID.PROTOSS_TEMPLARARCHIVE.value ], [], [] )
        self.m_upgradeData[UPGRADE_ID.WARPGATERESEARCH.value] =           TypeData( RACE.Protoss,  50,  50, 0, self.fps*160, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_WARPGATE.value, 0, [ UNIT_TYPEID.PROTOSS_CYBERNETICSCORE.value ], [], [] )

    # Zerg Upgrades
        self.m_upgradeData[UPGRADE_ID.BURROW.value] =                     TypeData( RACE.Zerg, 100, 100, 0, self.fps*100, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_BURROW.value, 0, [ UNIT_TYPEID.ZERG_HATCHERY.value, UNIT_TYPEID.ZERG_LAIR.value, UNIT_TYPEID.ZERG_HIVE.value ], [], [] )
        self.m_upgradeData[UPGRADE_ID.CENTRIFICALHOOKS.value] =           TypeData( RACE.Zerg, 150, 150, 0, self.fps*110, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_CENTRIFUGALHOOKS.value, 0, [ UNIT_TYPEID.ZERG_BANELINGNEST.value ], [ UNIT_TYPEID.ZERG_LAIR.value ], [] )
        self.m_upgradeData[UPGRADE_ID.CHITINOUSPLATING.value] =           TypeData( RACE.Zerg, 150, 150, 0, self.fps*110, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_CHITINOUSPLATING.value, 0, [ UNIT_TYPEID.ZERG_ULTRALISKCAVERN.value ], [], [] )
        self.m_upgradeData[UPGRADE_ID.EVOLVEMUSCULARAUGMENTS.value] =     TypeData( RACE.Zerg, 150, 150, 0, self.fps*100, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_MUSCULARAUGMENTS.value, 0, [ UNIT_TYPEID.ZERG_HYDRALISKDEN.value ], [], [] )
        self.m_upgradeData[UPGRADE_ID.GLIALRECONSTITUTION.value] =        TypeData( RACE.Zerg, 100, 100, 0, self.fps*110, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_GLIALREGENERATION.value, 0, [ UNIT_TYPEID.ZERG_ROACHWARREN.value ], [ UNIT_TYPEID.ZERG_LAIR.value, UNIT_TYPEID.ZERG_HIVE.value ], [] )
        self.m_upgradeData[UPGRADE_ID.INFESTORENERGYUPGRADE.value] =      TypeData( RACE.Zerg, 150, 150, 0, self.fps*80, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_PATHOGENGLANDS.value, 0, [ UNIT_TYPEID.ZERG_INFESTATIONPIT.value ], [], [] )
        self.m_upgradeData[UPGRADE_ID.NEURALPARASITE.value] =             TypeData( RACE.Zerg, 150, 150, 0, self.fps*110, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_NEURALPARASITE.value, 0, [ UNIT_TYPEID.ZERG_INFESTATIONPIT.value ], [], [] )
        self.m_upgradeData[UPGRADE_ID.OVERLORDSPEED.value] =              TypeData( RACE.Zerg, 100, 100, 0,  self.fps*60, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_PNEUMATIZEDCARAPACE.value, 0, [ UNIT_TYPEID.ZERG_HATCHERY.value, UNIT_TYPEID.ZERG_LAIR.value, UNIT_TYPEID.ZERG_HIVE.value ], [], [] )
        self.m_upgradeData[UPGRADE_ID.TUNNELINGCLAWS.value] =             TypeData( RACE.Zerg, 150, 150, 0, self.fps*110, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_TUNNELINGCLAWS.value, 0, [ UNIT_TYPEID.ZERG_ROACHWARREN.value ], [ UNIT_TYPEID.ZERG_LAIR.value, UNIT_TYPEID.ZERG_HIVE.value ], [] )
        self.m_upgradeData[UPGRADE_ID.ZERGFLYERARMORSLEVEL1.value] =      TypeData( RACE.Zerg, 150, 150, 0, self.fps*160, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_ZERGFLYERARMORLEVEL1.value, 0, [ UNIT_TYPEID.ZERG_SPIRE.value, UNIT_TYPEID.ZERG_GREATERSPIRE.value ], [], [] )
        self.m_upgradeData[UPGRADE_ID.ZERGFLYERARMORSLEVEL2.value] =      TypeData( RACE.Zerg, 225, 225, 0, self.fps*190, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_ZERGFLYERARMORLEVEL2.value, 0, [ UNIT_TYPEID.ZERG_SPIRE.value, UNIT_TYPEID.ZERG_GREATERSPIRE.value ], [ UNIT_TYPEID.ZERG_LAIR.value, UNIT_TYPEID.ZERG_HIVE.value ], [ UPGRADE_ID.ZERGFLYERARMORSLEVEL1.value] )
        self.m_upgradeData[UPGRADE_ID.ZERGFLYERARMORSLEVEL3.value] =      TypeData( RACE.Zerg, 300, 300, 0, self.fps*220, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_ZERGFLYERARMORLEVEL3.value, 0, [ UNIT_TYPEID.ZERG_SPIRE.value, UNIT_TYPEID.ZERG_GREATERSPIRE.value ], [ UNIT_TYPEID.ZERG_HIVE.value ], [ UPGRADE_ID.ZERGFLYERARMORSLEVEL2.value] )
        self.m_upgradeData[UPGRADE_ID.ZERGFLYERWEAPONSLEVEL1.value] =     TypeData( RACE.Zerg, 100, 100, 0, self.fps*160, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_ZERGFLYERATTACKLEVEL1.value, 0, [ UNIT_TYPEID.ZERG_SPIRE.value, UNIT_TYPEID.ZERG_GREATERSPIRE.value ], [], [] )
        self.m_upgradeData[UPGRADE_ID.ZERGFLYERWEAPONSLEVEL2.value] =     TypeData( RACE.Zerg, 175, 175, 0, self.fps*190, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_ZERGFLYERATTACKLEVEL2.value, 0, [ UNIT_TYPEID.ZERG_SPIRE.value, UNIT_TYPEID.ZERG_GREATERSPIRE.value ], [ UNIT_TYPEID.ZERG_LAIR.value, UNIT_TYPEID.ZERG_HIVE.value ], [ UPGRADE_ID.ZERGFLYERWEAPONSLEVEL1.value] )
        self.m_upgradeData[UPGRADE_ID.ZERGFLYERWEAPONSLEVEL3.value] =     TypeData( RACE.Zerg, 250, 250, 0, self.fps*220, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_ZERGFLYERATTACKLEVEL3.value, 0, [ UNIT_TYPEID.ZERG_SPIRE.value, UNIT_TYPEID.ZERG_GREATERSPIRE.value ], [ UNIT_TYPEID.ZERG_HIVE.value ], [ UPGRADE_ID.ZERGFLYERWEAPONSLEVEL2.value] )
        self.m_upgradeData[UPGRADE_ID.ZERGGROUNDARMORSLEVEL1.value] =     TypeData( RACE.Zerg, 150, 150, 0, self.fps*160, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_ZERGGROUNDARMORLEVEL1.value, 0, [ UNIT_TYPEID.ZERG_EVOLUTIONCHAMBER.value ], [], [] )
        self.m_upgradeData[UPGRADE_ID.ZERGGROUNDARMORSLEVEL2.value] =     TypeData( RACE.Zerg, 225, 225, 0, self.fps*190, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_ZERGGROUNDARMORLEVEL2.value, 0, [ UNIT_TYPEID.ZERG_EVOLUTIONCHAMBER.value ], [ UNIT_TYPEID.ZERG_LAIR.value, UNIT_TYPEID.ZERG_HIVE.value ], [UPGRADE_ID.ZERGGROUNDARMORSLEVEL1.value] )
        self.m_upgradeData[UPGRADE_ID.ZERGGROUNDARMORSLEVEL3.value] =     TypeData( RACE.Zerg, 300, 300, 0, self.fps*220, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_ZERGGROUNDARMORLEVEL3.value, 0, [ UNIT_TYPEID.ZERG_EVOLUTIONCHAMBER.value ], [ UNIT_TYPEID.ZERG_HIVE.value ], [UPGRADE_ID.ZERGGROUNDARMORSLEVEL2.value] )
        self.m_upgradeData[UPGRADE_ID.ZERGLINGATTACKSPEED.value] =        TypeData( RACE.Zerg, 200, 200, 0, self.fps*130, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_ZERGLINGADRENALGLANDS.value, 0, [ UNIT_TYPEID.ZERG_SPAWNINGPOOL.value ], [ UNIT_TYPEID.ZERG_HIVE.value ], [] )
        self.m_upgradeData[UPGRADE_ID.ZERGLINGMOVEMENTSPEED.value] =      TypeData( RACE.Zerg, 100, 100, 0, self.fps*110, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_ZERGLINGMETABOLICBOOST.value, 0, [ UNIT_TYPEID.ZERG_SPAWNINGPOOL.value ], [], [] )
        self.m_upgradeData[UPGRADE_ID.ZERGMELEEWEAPONSLEVEL1.value] =     TypeData( RACE.Zerg, 100, 100, 0, self.fps*160, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_ZERGMELEEWEAPONSLEVEL1.value, 0, [ UNIT_TYPEID.ZERG_EVOLUTIONCHAMBER.value ], [], [] )
        self.m_upgradeData[UPGRADE_ID.ZERGMELEEWEAPONSLEVEL2.value] =     TypeData( RACE.Zerg, 150, 150, 0, self.fps*190, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_ZERGMELEEWEAPONSLEVEL2.value, 0, [ UNIT_TYPEID.ZERG_EVOLUTIONCHAMBER.value ], [ UNIT_TYPEID.ZERG_LAIR.value, UNIT_TYPEID.ZERG_HIVE.value ], [UPGRADE_ID.ZERGMELEEWEAPONSLEVEL1.value] )
        self.m_upgradeData[UPGRADE_ID.ZERGMELEEWEAPONSLEVEL3.value] =     TypeData( RACE.Zerg, 200, 200, 0, self.fps*220, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_ZERGMELEEWEAPONSLEVEL3.value, 0, [ UNIT_TYPEID.ZERG_EVOLUTIONCHAMBER.value ], [ UNIT_TYPEID.ZERG_HIVE.value ], [UPGRADE_ID.ZERGMELEEWEAPONSLEVEL2.value] )
        self.m_upgradeData[UPGRADE_ID.ZERGMISSILEWEAPONSLEVEL1.value] =   TypeData( RACE.Zerg, 100, 100, 0, self.fps*160, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_ZERGMISSILEWEAPONSLEVEL1.value, 0, [ UNIT_TYPEID.ZERG_EVOLUTIONCHAMBER.value ], [], [] )
        self.m_upgradeData[UPGRADE_ID.ZERGMISSILEWEAPONSLEVEL2.value] =   TypeData( RACE.Zerg, 150, 150, 0, self.fps*190, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_ZERGMISSILEWEAPONSLEVEL2.value, 0, [ UNIT_TYPEID.ZERG_EVOLUTIONCHAMBER.value ], [ UNIT_TYPEID.ZERG_LAIR.value, UNIT_TYPEID.ZERG_HIVE.value ], [UPGRADE_ID.ZERGMISSILEWEAPONSLEVEL1.value] )
        self.m_upgradeData[UPGRADE_ID.ZERGMISSILEWEAPONSLEVEL3.value] =   TypeData( RACE.Zerg, 200, 200, 0, self.fps*220, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_ZERGMISSILEWEAPONSLEVEL3.value, 0, [ UNIT_TYPEID.ZERG_EVOLUTIONCHAMBER.value ], [ UNIT_TYPEID.ZERG_HIVE.value ], [UPGRADE_ID.ZERGMISSILEWEAPONSLEVEL2.value] )

    def updateUnitTypeData(self, version):
        if distutils.version.LooseVersion(version) >= distutils.version.LooseVersion('4.0'):
            self.m_unitTypeData[UNIT_TYPEID.ZERG_LURKERDENMP.value] = TypeData(RACE.Zerg, 100, 150, 0, self.fps*25, True, True, False, False, False, False, False, ABILITY_ID.BUILD_LURKERDENMP.value, 0, [ UNIT_TYPEID.ZERG_DRONE.value ], [ UNIT_TYPEID.ZERG_HYDRALISKDEN.value ], [])
            data_raw = data_raw_4_0
            for unit_type in self.m_unitTypeData:
                mineralCost = data_raw.units[unit_type].mineral_cost
                gasCost = data_raw.units[unit_type].vespene_cost
                supplyCost = data_raw.units[unit_type].food_required
                buildTime = data_raw.units[unit_type].build_time
                if self.m_unitTypeData[unit_type].race == RACE.Zerg:
                    if (unit_type not in [UNIT_TYPEID.ZERG_LARVA.value,
                                          UNIT_TYPEID.ZERG_EGG.value,
                                          UNIT_TYPEID.ZERG_QUEEN.value,
                                          UNIT_TYPEID.ZERG_NYDUSCANAL.value,
                                          UNIT_TYPEID.ZERG_ZERGLING.value]):
                        builder_type = self.m_unitTypeData[unit_type].whatBuilds[0]
                        mineralCost -= data_raw.units[builder_type].mineral_cost
                        gasCost -= data_raw.units[builder_type].vespene_cost
                    if (unit_type == UNIT_TYPEID.ZERG_ZERGLING.value):
                        mineralCost *= 2
                        gasCost *= 2
                        supplyCost *= 2
                    if (unit_type in [UNIT_TYPEID.ZERG_BANELING.value,
                                          UNIT_TYPEID.ZERG_BROODLORD.value,
                                          UNIT_TYPEID.ZERG_RAVAGER.value,
                                          UNIT_TYPEID.ZERG_LURKERMP.value,
                                          UNIT_TYPEID.ZERG_OVERSEER.value]):
                        builder_type = self.m_unitTypeData[unit_type].whatBuilds[0]
                        supplyCost -= data_raw.units[builder_type].food_required
                    #if (mineralCost != self.m_unitTypeData[unit_type].mineralCost
                    #        or gasCost != self.m_unitTypeData[unit_type].gasCost
                    #        or supplyCost != self.m_unitTypeData[unit_type].supplyCost
                    #        or buildTime != self.m_unitTypeData[unit_type].buildTime):
                    #    print('Data for unit {} inconsistent!'.format(unit_type))
                    self.m_unitTypeData[unit_type].mineralCost = mineralCost
                    self.m_unitTypeData[unit_type].gasCost = gasCost
                    self.m_unitTypeData[unit_type].supplyCost = supplyCost
                    self.m_unitTypeData[unit_type].buildTime = buildTime

    def updateUpgradeData(self, version):
        if distutils.version.LooseVersion(version) >= distutils.version.LooseVersion('4.1.4'):
            self.m_upgradeData[UPGRADE_ID.EVOLVEGROOVEDSPINES.value] = TypeData(RACE.Zerg, 100, 100, 0, self.fps * 100, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_GROOVEDSPINES.value, 0, [UNIT_TYPEID.ZERG_HYDRALISKDEN.value], [], [])
            self.m_upgradeData[UPGRADE_ID.EVOLVEMUSCULARAUGMENTS.value] = TypeData(RACE.Zerg, 100, 100, 0, self.fps * 100, False, False, False, False, False, False, False, ABILITY_ID.RESEARCH_MUSCULARAUGMENTS.value, 0, [UNIT_TYPEID.ZERG_HYDRALISKDEN.value], [], [])

        if distutils.version.LooseVersion(version) >= distutils.version.LooseVersion('4.0'):
            data_raw = data_raw_4_0
            for upgrade_type in self.m_upgradeData:
                mineralCost = data_raw.upgrades[upgrade_type].mineral_cost
                gasCost = data_raw.upgrades[upgrade_type].vespene_cost
                buildTime = data_raw.upgrades[upgrade_type].research_time
                if (mineralCost != 0): # 0 means upgrade is removed from 4.0
                    #if (mineralCost != self.m_upgradeData[upgrade_type].mineralCost
                    #        or gasCost != self.m_upgradeData[upgrade_type].gasCost
                    #        or buildTime != self.m_upgradeData[upgrade_type].buildTime):
                    #    print('Data for upgrade {} inconsistent!'.format(upgrade_type))
                    self.m_upgradeData[upgrade_type].mineralCost = mineralCost
                    self.m_upgradeData[upgrade_type].gasCost = gasCost
                    self.m_upgradeData[upgrade_type].buildTime = buildTime

if __name__ == "__main__":
  TT = TechTree()
  TT.updateUnitTypeData('4.0.2')
