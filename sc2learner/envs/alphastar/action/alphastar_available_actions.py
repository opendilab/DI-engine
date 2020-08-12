import collections
import torch
from pysc2.lib import features
from pysc2.lib.static_data import NUM_ACTIONS, ACTIONS_REORDER, UPGRADES_REORDER_INV

Avail_fn = collections.namedtuple(
    'Avail_fn', ['func_id', 'func_name', 'units', 'upgrade', 'resource', 'mana', 'supply']
)

FUNCTION_LIST = [
    Avail_fn(0, "no_op", None, None, [], 0, 0),
    Avail_fn(1, "Smart_pt", None, None, [], 0, 0),
    Avail_fn(2, "Attack_pt", None, None, [], 0, 0),
    Avail_fn(3, "Attack_unit", None, None, [], 0, 0),
    Avail_fn(12, "Smart_unit", None, None, [], 0, 0),
    Avail_fn(13, "Move_pt", None, None, [], 0, 0),
    Avail_fn(14, "Move_unit", None, None, [], 0, 0),
    Avail_fn(15, "Patrol_pt", None, None, [], 0, 0),
    Avail_fn(16, "Patrol_unit", None, None, [], 0, 0),
    Avail_fn(17, "HoldPosition_quick", None, None, [], 0, 0),
    Avail_fn(18, "Research_InterceptorGravitonCatapult_quick", [[64]], None, [150, 150], 0, 0),
    Avail_fn(19, "Research_PhoenixAnionPulseCrystals_quick", [[64]], None, [150, 150], 0, 0),
    Avail_fn(20, "Effect_GuardianShield_quick", [[77]], None, [], 75, 0),
    Avail_fn(21, "Train_Mothership_quick", [[59], [64]], None, [400, 400], 0, 0),
    Avail_fn(22, "Hallucination_Archon_quick", [[77]], None, [], 75, 0),
    Avail_fn(23, "Hallucination_Colossus_quick", [[77]], None, [], 75, 0),
    Avail_fn(24, "Hallucination_HighTemplar_quick", [[77]], None, [], 75, 0),
    Avail_fn(25, "Hallucination_Immortal_quick", [[77]], None, [], 75, 0),
    Avail_fn(26, "Hallucination_Phoenix_quick", [[77]], None, [], 75, 0),
    Avail_fn(27, "Hallucination_Probe_quick", [[77]], None, [], 75, 0),
    Avail_fn(28, "Hallucination_Stalker_quick", [[77]], None, [], 75, 0),
    Avail_fn(29, "Hallucination_VoidRay_quick", [[77]], None, [], 75, 0),
    Avail_fn(30, "Hallucination_WarpPrism_quick", [[77]], None, [], 75, 0),
    Avail_fn(31, "Hallucination_Zealot_quick", [[77]], None, [], 75, 0),
    Avail_fn(32, "Effect_GravitonBeam_unit", [[78]], None, [], 50, 0),
    Avail_fn(33, "Effect_ChronoBoost_unit", [[59]], None, [], 50, 0),
    Avail_fn(34, "Build_Nexus_pt", [[84]], None, [400, 0], 0, 0),
    Avail_fn(35, "Build_Pylon_pt", [[84]], None, [100, 0], 0, 0),
    Avail_fn(36, "Build_Assimilator_unit", [[84]], None, [25, 0], 0, 0),
    Avail_fn(37, "Build_Gateway_pt", [[84], [59]], None, [150, 0], 0, 0),
    Avail_fn(38, "Build_Forge_pt", [[84], [59]], None, [150, 0], 0, 0),
    Avail_fn(39, "Build_FleetBeacon_pt", [[84], [67]], None, [300, 200], 0, 0),
    Avail_fn(40, "Build_TwilightCouncil_pt", [[84], [72]], None, [150, 100], 0, 0),
    Avail_fn(41, "Build_PhotonCannon_pt", [[84], [63]], None, [150, 0], 0, 0),
    Avail_fn(42, "Build_Stargate_pt", [[84], [72]], None, [150, 150], 0, 0),
    Avail_fn(43, "Build_TemplarArchive_pt", [[84], [65]], None, [150, 200], 0, 0),
    Avail_fn(44, "Build_DarkShrine_pt", [[84], [65]], None, [150, 150], 0, 0),
    Avail_fn(45, "Build_RoboticsBay_pt", [[84], [71]], None, [150, 150], 0, 0),
    Avail_fn(46, "Build_RoboticsFacility_pt", [[84], [72]], None, [200, 100], 0, 0),
    Avail_fn(47, "Build_CyberneticsCore_pt", [[84], [62, 133]], None, [150, 0], 0, 0),
    Avail_fn(48, "Build_ShieldBattery_pt", [[84], [72]], None, [100, 0], 0, 0),
    Avail_fn(49, "Train_Zealot_quick", [[62]], None, [100, 0], 0, 0),
    Avail_fn(50, "Train_Stalker_quick", [[62], [72]], None, [125, 50], 0, 0),
    Avail_fn(51, "Train_HighTemplar_quick", [[62], [68]], None, [50, 150], 0, 0),
    Avail_fn(52, "Train_DarkTemplar_quick", [[62], [69]], None, [125, 125], 0, 0),
    Avail_fn(53, "Train_Sentry_quick", [[62], [72]], None, [50, 100], 0, 0),
    Avail_fn(54, "Train_Adept_quick", [[62], [72]], None, [100, 25], 0, 0),
    Avail_fn(55, "Train_Phoenix_quick", [[67]], None, [150, 100], 0, 0),
    Avail_fn(56, "Train_Carrier_quick", [[67], [64]], None, [350, 250], 0, 0),
    Avail_fn(57, "Train_VoidRay_quick", [[67]], None, [250, 150], 0, 0),
    Avail_fn(58, "Train_Oracle_quick", [[67]], None, [150, 150], 0, 0),
    Avail_fn(59, "Train_Tempest_quick", [[67], [64]], None, [250, 175], 0, 0),
    Avail_fn(60, "Train_WarpPrism_quick", [[71]], None, [250, 0], 0, 0),
    Avail_fn(61, "Train_Observer_quick", [[71]], None, [25, 75], 0, 0),
    Avail_fn(62, "Train_Colossus_quick", [[71], [70]], None, [300, 200], 0, 0),
    Avail_fn(63, "Train_Immortal_quick", [[71]], None, [275, 100], 0, 0),
    Avail_fn(64, "Train_Probe_quick", [[59]], None, [50, 0], 0, 0),
    Avail_fn(65, "Effect_PsiStorm_pt", [[75]], 52, [], 75, 0),
    Avail_fn(66, "Build_Interceptors_quick", [[79]], None, [15, 0], 0, 0),
    Avail_fn(67, "Research_GraviticBooster_quick", [[70]], None, [100, 100], 0, 0),
    Avail_fn(68, "Research_GraviticDrive_quick", [[70]], None, [100, 100], 0, 0),
    Avail_fn(69, "Research_ExtendedThermalLance_quick", [[70]], None, [150, 150], 0, 0),
    Avail_fn(70, "Research_PsiStorm_quick", [[68]], None, [200, 200], 0, 0),
    Avail_fn(71, "TrainWarp_Zealot_pt", [[133]], None, [100, 0], 0, 2),
    Avail_fn(72, "TrainWarp_Stalker_pt", [[133]], None, [125, 50], 0, 2),
    Avail_fn(73, "TrainWarp_HighTemplar_pt", [[133]], None, [50, 150], 0, 2),
    Avail_fn(74, "TrainWarp_DarkTemplar_pt", [[133]], None, [125, 125], 0, 2),
    Avail_fn(75, "TrainWarp_Sentry_pt", [[133]], None, [50, 100], 0, 2),
    Avail_fn(76, "TrainWarp_Adept_pt", [[133]], None, [100, 25], 0, 2),
    Avail_fn(77, "Morph_WarpGate_quick", [[62]], 84, [], 0, 0),
    Avail_fn(78, "Morph_Gateway_quick", [[133]], None, [], 0, 0),
    Avail_fn(79, "Effect_ForceField_pt", [[77]], None, [], 50, 0),
    Avail_fn(80, "Morph_WarpPrismPhasingMode_quick", [[81]], None, [], 0, 0),
    Avail_fn(81, "Morph_WarpPrismTransportMode_quick", [[136]], None, [], 0, 0),
    Avail_fn(82, "Research_WarpGate_quick", [[72]], None, [50, 50], 0, 0),
    Avail_fn(83, "Research_Charge_quick", [[65]], None, [100, 100], 0, 0),
    Avail_fn(84, "Research_Blink_quick", [[65]], None, [150, 150], 0, 0),
    Avail_fn(85, "Research_AdeptResonatingGlaives_quick", [[65]], None, [100, 100], 0, 0),
    Avail_fn(86, "Morph_Archon_quick", [[75, 76]], None, [], 0, 0),
    Avail_fn(87, "Behavior_BuildingAttackOn_quick", [[9]], None, [], 0, 0),
    Avail_fn(88, "Behavior_BuildingAttackOff_quick", [[9]], None, [], 0, 0),
    Avail_fn(89, "Hallucination_Oracle_quick", [[77]], None, [], 75, 0),
    Avail_fn(90, "Effect_OracleRevelation_pt", [[495]], None, [], 50, 0),
    Avail_fn(91, "Effect_ImmortalBarrier_quick", [[83]], None, [], 0, 0),
    Avail_fn(92, "Hallucination_Disruptor_quick", [[77]], None, [], 75, 0),
    Avail_fn(93, "Hallucination_Adept_quick", [[77]], None, [], 75, 0),
    Avail_fn(94, "Effect_VoidRayPrismaticAlignment_quick", [[80]], None, [], 0, 0),
    Avail_fn(95, "Build_StasisTrap_pt", [[495]], None, [], 50, 0),
    Avail_fn(96, "Effect_AdeptPhaseShift_pt", [[311]], None, [], 0, 0),
    Avail_fn(97, "Research_ShadowStrike_quick", [[69]], None, [100, 100], 0, 0),
    Avail_fn(98, "Cancel_quick", None, None, [], 0, 0),
    Avail_fn(99, "Halt_quick", None, None, [], 0, 0),
    Avail_fn(100, "UnloadAll_quick", [[81, 136, 24, 18, 54, 142, 95, 893]], None, [], 0, 0),
    Avail_fn(101, "Stop_quick", None, None, [], 0, 0),
    Avail_fn(102, "Harvest_Gather_unit", None, None, [], 0, 0),
    Avail_fn(103, "Harvest_Return_quick", None, None, [], 0, 0),
    Avail_fn(104, "Load_unit", [[81, 136, 24, 18, 54, 142, 95, 893]], None, [], 0, 0),
    Avail_fn(105, "UnloadAllAt_pt", [[81, 136, 24, 18, 54, 142, 95, 893]], None, [], 0, 0),
    Avail_fn(106, "Rally_Units_pt", None, None, [], 0, 0),
    Avail_fn(107, "Rally_Units_unit", None, None, [], 0, 0),
    Avail_fn(108, "Effect_Repair_pt", [[45, 268]], None, [], 0, 0),
    Avail_fn(109, "Effect_Repair_unit", [[45, 268]], None, [], 0, 0),
    Avail_fn(110, "Effect_MassRecall_pt", [[59, 488]], None, [], 50, 0),
    Avail_fn(111, "Effect_Blink_pt", [[74]], 87, [], 0, 0),
    Avail_fn(112, "Effect_Blink_unit", [[74]], 87, [], 0, 0),
    Avail_fn(114, "Rally_Workers_pt", None, None, [], 0, 0),
    Avail_fn(115, "Rally_Workers_unit", None, None, [], 0, 0),
    Avail_fn(116, "Research_ProtossAirArmor_quick", [[72]], None, [150, 150], 0, 0),
    Avail_fn(117, "Research_ProtossAirWeapons_quick", [[72]], None, [100, 100], 0, 0),
    Avail_fn(118, "Research_ProtossGroundArmor_quick", [[63]], None, [100, 100], 0, 0),
    Avail_fn(119, "Research_ProtossGroundWeapons_quick", [[63]], None, [100, 100], 0, 0),
    Avail_fn(120, "Research_ProtossShields_quick", [[63]], None, [150, 150], 0, 0),
    Avail_fn(121, "Morph_ObserverMode_quick", [[1911]], None, [], 0, 0),
    Avail_fn(122, "Effect_ChronoBoostEnergyCost_unit", [[59]], None, [], 50, 0),
    Avail_fn(129, "Cancel_Last_quick", None, None, [], 0, 0),
    Avail_fn(157, "Effect_Feedback_unit", [[75]], None, [], 50, 0),
    Avail_fn(158, "Behavior_PulsarBeamOff_quick", [[495]], None, [], 0, 0),
    Avail_fn(159, "Behavior_PulsarBeamOn_quick", [[495]], None, [], 25, 0),
    Avail_fn(160, "Morph_SurveillanceMode_quick", [[82]], None, [], 0, 0),
    Avail_fn(161, "Effect_Restore_unit", [[1910]], None, [], 0, 0),
    Avail_fn(164, "UnloadAllAt_unit", [[81, 136, 24, 18, 54, 142, 95, 893]], None, [], 0, 0),
    Avail_fn(166, "Train_Disruptor_quick", [[70]], None, [150, 150], 0, 0),
    Avail_fn(167, "Effect_PurificationNova_pt", [[694]], None, [], 0, 0),
    Avail_fn(168, "raw_move_camera", None, None, [], 0, 0),
    Avail_fn(169, "Behavior_CloakOff_quick", [[50, 55, 144]], None, [], 0, 0),
    Avail_fn(172, "Behavior_CloakOn_quick", [[50, 55, 144]], None, [], 25, 0),
    Avail_fn(175, "Behavior_GenerateCreepOff_quick", [[100, 101]], None, [], 0, 0),
    Avail_fn(176, "Behavior_GenerateCreepOn_quick", [[100, 101]], None, [], 0, 0),
    Avail_fn(177, "Behavior_HoldFireOff_quick", [[503, 50, 144]], None, [], 0, 0),
    Avail_fn(180, "Behavior_HoldFireOn_quick", [[503, 50, 144]], None, [], 0, 0),
    Avail_fn(183, "Build_Armory_pt", [[45], [27, 43]], None, [150, 100], 0, 0),
    Avail_fn(184, "Build_BanelingNest_pt", [[89], [104]], None, [100, 50], 0, 0),
    Avail_fn(185, "Build_Barracks_pt", [[45], [19, 47]], None, [150, 0], 0, 0),
    Avail_fn(186, "Build_Bunker_pt", [[45], [21, 46]], None, [100, 0], 0, 0),
    Avail_fn(187, "Build_CommandCenter_pt", [[45]], None, [400, 0], 0, 0),
    Avail_fn(188, "Build_CreepTumor_pt", [[87, 126, 137]], None, [], 0, 0),
    Avail_fn(191, "Build_EngineeringBay_pt", [[45], [18, 36, 132, 134, 130]], None, [125, 0], 0, 0),
    Avail_fn(192, "Build_EvolutionChamber_pt", [[86, 100, 101], [104]], None, [75, 0], 0, 0),
    Avail_fn(193, "Build_Extractor_unit", [[104]], None, [25, 0], 0, 0),
    Avail_fn(194, "Build_Factory_pt", [[21, 46], [45]], None, [150, 100], 0, 0),
    Avail_fn(195, "Build_FusionCore_pt", [[28, 44], [45]], None, [150, 150], 0, 0),
    Avail_fn(196, "Build_GhostAcademy_pt", [[21, 46], [45]], None, [150, 50], 0, 0),
    Avail_fn(197, "Build_Hatchery_pt", [[104]], None, [300, 0], 0, 0),
    Avail_fn(198, "Build_HydraliskDen_pt", [[100, 101], [104]], None, [100, 100], 0, 0),
    Avail_fn(199, "Build_InfestationPit_pt", [[100, 101], [104]], None, [100, 100], 0, 0),
    Avail_fn(200, "Build_Interceptors_autocast", [[79]], None, [], 0, 0),
    Avail_fn(201, "Build_LurkerDen_pt", [[91], [104], [100, 101]], None, [100, 150], 0, 0),
    Avail_fn(202, "Build_MissileTurret_pt", [[45], [22]], None, [100, 0], 0, 0),
    Avail_fn(203, "Build_Nuke_quick", [[26]], None, [100, 100], 0, 0),
    Avail_fn(204, "Build_NydusNetwork_pt", [[100, 101], [104]], None, [150, 150], 0, 0),
    Avail_fn(205, "Build_NydusWorm_pt", [[95]], None, [50, 50], 0, 0),
    Avail_fn(206, "Build_Reactor_quick", [[21, 46, 27, 43, 28, 44]], None, [50, 50], 0, 0),
    Avail_fn(207, "Build_Reactor_pt", [[21, 46, 27, 43, 28, 44]], None, [50, 50], 0, 0),
    Avail_fn(214, "Build_Refinery_pt", [[45]], None, [75, 0], 0, 0),
    Avail_fn(215, "Build_RoachWarren_pt", [[89], [104]], None, [150, 0], 0, 0),
    Avail_fn(216, "Build_SensorTower_pt", [[45], [22]], None, [125, 100], 0, 0),
    Avail_fn(217, "Build_SpawningPool_pt", [[86, 100, 101], [104]], None, [200, 0], 0, 0),
    Avail_fn(218, "Build_SpineCrawler_pt", [[89], [104]], None, [100, 0], 0, 0),
    Avail_fn(219, "Build_Spire_pt", [[100, 101], [104]], None, [200, 200], 0, 0),
    Avail_fn(220, "Build_SporeCrawler_pt", [[89], [104]], None, [75, 0], 0, 0),
    Avail_fn(221, "Build_Starport_pt", [[27, 43], [45]], None, [150, 100], 0, 0),
    Avail_fn(222, "Build_SupplyDepot_pt", [[45]], None, [100, 0], 0, 0),
    Avail_fn(223, "Build_TechLab_quick", [[21, 46, 27, 43, 28, 44]], None, [50, 25], 0, 0),
    Avail_fn(224, "Build_TechLab_pt", [[21, 46, 27, 43, 28, 44]], None, [50, 25], 0, 0),
    Avail_fn(231, "Build_UltraliskCavern_pt", [[101], [104]], None, [150, 200], 0, 0),
    Avail_fn(232, "BurrowDown_quick", None, None, [], 0, 0),
    Avail_fn(246, "BurrowUp_quick", None, None, [], 0, 0),
    Avail_fn(247, "BurrowUp_autocast", 0, 0, [], 0, 0),
    Avail_fn(293, "Effect_Abduct_unit", [[499]], None, [], 75, 0),
    Avail_fn(294, "Effect_AntiArmorMissile_unit", [[56]], None, [], 75, 0),
    Avail_fn(295, "Effect_AutoTurret_pt", [[56]], None, [], 50, 0),
    Avail_fn(296, "Effect_BlindingCloud_pt", [[499]], None, [], 100, 0),
    Avail_fn(297, "Effect_CalldownMULE_pt", [[132]], None, [], 50, 0),
    Avail_fn(298, "Effect_CalldownMULE_unit", [[132]], None, [], 50, 0),
    Avail_fn(299, "Effect_CausticSpray_unit", [[112]], None, [], 0, 0),
    Avail_fn(300, "Effect_Charge_pt", [[73]], 86, [], 0, 0),
    Avail_fn(301, "Effect_Charge_unit", [[73]], 86, [], 0, 0),
    Avail_fn(302, "Effect_Charge_autocast", [[73]], 86, [], 0, 0),
    Avail_fn(303, "Effect_Contaminate_unit", [[129]], None, [], 125, 0),
    Avail_fn(304, "Effect_CorrosiveBile_pt", [[688]], None, [], 0, 0),
    Avail_fn(305, "Effect_EMP_pt", [[50, 144]], None, [], 75, 0),
    Avail_fn(306, "Effect_EMP_unit", [[50, 144]], None, [], 75, 0),
    Avail_fn(307, "Effect_Explode_quick", [[9, 115]], None, [], 0, 0),
    Avail_fn(308, "Effect_FungalGrowth_pt", [[111]], None, [], 75, 0),
    Avail_fn(309, "Effect_FungalGrowth_unit", [[111]], None, [], 75, 0),
    Avail_fn(310, "Effect_GhostSnipe_unit", [[50, 144]], None, [], 25, 0),
    Avail_fn(311, "Effect_Heal_unit", [[54]], None, [], 0, 0),
    Avail_fn(312, "Effect_Heal_autocast", [[54]], None, [], 0, 0),
    Avail_fn(313, "Effect_ImmortalBarrier_autocast", [[83]], None, [], 0, 0),
    Avail_fn(314, "Effect_InfestedTerrans_pt", [[111, 127]], None, [], 25, 0),
    Avail_fn(315, "Effect_InjectLarva_unit", [[126], [86, 100, 101]], None, [], 25, 0),
    Avail_fn(316, "Effect_InterferenceMatrix_unit", [[56]], None, [], 75, 0),
    Avail_fn(317, "Effect_KD8Charge_pt", [[49]], None, [], 0, 0),
    Avail_fn(318, "Effect_LockOn_unit", [[692]], None, [], 0, 0),
    Avail_fn(319, "Effect_LocustSwoop_pt", [[693]], None, [], 0, 0),
    Avail_fn(320, "Effect_MedivacIgniteAfterburners_quick", [[54]], None, [], 0, 0),
    Avail_fn(321, "Effect_NeuralParasite_unit", [[111, 127]], None, [], 100, 0),
    Avail_fn(322, "Effect_NukeCalldown_pt", [[50, 144]], None, [], 0, 0),
    Avail_fn(323, "Effect_ParasiticBomb_unit", [[499]], None, [], 125, 0),
    Avail_fn(324, "Effect_Repair_autocast", [[45, 268]], None, [], 0, 0),
    Avail_fn(331, "Effect_Restore_autocast", [[1910]], None, [], 0, 0),
    Avail_fn(332, "Effect_Salvage_quick", [[24]], None, [], 0, 0),
    Avail_fn(333, "Effect_Scan_pt", [[132]], None, [], 50, 0),
    Avail_fn(334, "Effect_SpawnChangeling_quick", [[129, 1912]], None, [], 50, 0),
    Avail_fn(335, "Effect_SpawnLocusts_pt", [[494, 493]], None, [], 0, 0),
    Avail_fn(336, "Effect_SpawnLocusts_unit", [[494, 493]], None, [], 0, 0),
    Avail_fn(337, "Effect_Spray_pt", [[45, 84, 104]], None, [], 0, 0),
    Avail_fn(341, "Effect_Stim_quick", [[51, 48]], 15, [], 0, 0),
    Avail_fn(346, "Effect_SupplyDrop_unit", [[132]], None, [], 0, 0),
    Avail_fn(347, "Effect_TacticalJump_pt", [[57]], None, [], 0, 0),
    Avail_fn(348, "Effect_TimeWarp_pt", [[10, 488]], None, [], 100, 0),
    Avail_fn(349, "Effect_Transfusion_unit", [[126]], None, [], 50, 0),
    Avail_fn(350, "Effect_ViperConsume_unit", [[499]], None, [], 0, 0),
    Avail_fn(351, "Effect_WidowMineAttack_pt", [[500, 498]], None, [], 0, 0),
    Avail_fn(352, "Effect_WidowMineAttack_unit", [[500, 498]], None, [], 0, 0),
    Avail_fn(353, "Effect_WidowMineAttack_autocast", [[500, 498]], None, [], 0, 0),
    Avail_fn(363, "Land_pt", [[46, 36, 43, 134, 44]], None, [], 0, 0),
    Avail_fn(369, "Lift_quick", [[18, 27, 21, 132, 28]], None, [], 0, 0),
    Avail_fn(375, "LoadAll_quick", [[81, 136, 24, 18, 54, 142, 95, 893]], None, [], 0, 0),
    Avail_fn(383, "Morph_BroodLord_quick", [[102], [112]], None, [150, 150], 0, 2),
    Avail_fn(384, "Morph_GreaterSpire_quick", [[92], [101]], None, [100, 150], 0, 0),
    Avail_fn(385, "Morph_Hellbat_quick", [[53]], None, [], 0, 0),
    Avail_fn(386, "Morph_Hellion_quick", [[484]], None, [], 0, 0),
    Avail_fn(387, "Morph_Hive_quick", [[100], [94]], None, [200, 150], 0, 0),
    Avail_fn(388, "Morph_Lair_quick", [[86], [89]], None, [150, 100], 0, 0),
    Avail_fn(389, "Morph_LiberatorAAMode_quick", [[734]], None, [], 0, 0),
    Avail_fn(390, "Morph_LiberatorAGMode_pt", [[689]], None, [], 0, 0),
    Avail_fn(391, "Morph_Lurker_quick", [[107], [504]], None, [50, 100], 0, 1),
    Avail_fn(392, "Morph_LurkerDen_quick", [[91]], None, [], 0, 0),
    Avail_fn(393, "Morph_Mothership_quick", [[59], [64]], None, [], 0, 0),
    Avail_fn(394, "Morph_OrbitalCommand_quick", [[18]], None, [150, 0], 0, 0),
    Avail_fn(395, "Morph_OverlordTransport_quick", [[106], [100, 101]], None, [25, 25], 0, 0),
    Avail_fn(396, "Morph_Overseer_quick", [[100, 101], [106]], None, [50, 50], 0, 0),
    Avail_fn(397, "Morph_OverseerMode_quick", [[1912]], None, [], 0, 0),
    Avail_fn(398, "Morph_OversightMode_quick", [[129]], None, [], 0, 0),
    Avail_fn(399, "Morph_PlanetaryFortress_quick", [[18], [22]], None, [150, 150], 0, 0),
    Avail_fn(400, "Morph_Ravager_quick", [[110], [97]], None, [25, 75], 0, 1),
    Avail_fn(401, "Morph_Root_pt", [[139, 140, 98, 99]], None, [], 0, 0),
    Avail_fn(402, "Morph_SiegeMode_quick", [[33]], None, [], 0, 0),
    Avail_fn(407, "Morph_SupplyDepot_Lower_quick", [[19]], None, [], 0, 0),
    Avail_fn(408, "Morph_SupplyDepot_Raise_quick", [[47]], None, [], 0, 0),
    Avail_fn(409, "Morph_ThorExplosiveMode_quick", [[691]], None, [], 0, 0),
    Avail_fn(410, "Morph_ThorHighImpactMode_quick", [[52]], None, [], 0, 0),
    Avail_fn(411, "Morph_Unsiege_quick", [[32]], None, [], 0, 0),
    Avail_fn(412, "Morph_Uproot_quick", [[98, 99, 139, 140]], None, [], 0, 0),
    Avail_fn(413, "Morph_VikingAssaultMode_quick", [[35]], None, [], 0, 0),
    Avail_fn(414, "Morph_VikingFighterMode_quick", [[34]], None, [], 0, 0),
    Avail_fn(425, "Research_AdaptiveTalons_quick", [[101], [504]], None, [150, 150], 0, 0),
    Avail_fn(426, "Research_AdvancedBallistics_quick", [[30]], None, [150, 150], 0, 0),
    Avail_fn(427, "Research_BansheeCloakingField_quick", [[41]], None, [100, 100], 0, 0),
    Avail_fn(428, "Research_BansheeHyperflightRotors_quick", [[41]], None, [150, 150], 0, 0),
    Avail_fn(429, "Research_BattlecruiserWeaponRefit_quick", [[30]], None, [], 0, 0),
    Avail_fn(430, "Research_Burrow_quick", [[86, 100, 101, 498]], None, [100, 100], 0, 0),
    Avail_fn(431, "Research_CentrifugalHooks_quick", [[100, 101], [96]], None, [150, 150], 0, 0),
    Avail_fn(432, "Research_ChitinousPlating_quick", [[93]], None, [150, 150], 0, 0),
    Avail_fn(433, "Research_CombatShield_quick", [[37]], None, [100, 100], 0, 0),
    Avail_fn(434, "Research_ConcussiveShells_quick", [[37]], None, [50, 50], 0, 0),
    Avail_fn(435, "Research_CycloneRapidFireLaunchers_quick", [[39]], None, [75, 75], 0, 0),
    Avail_fn(436, "Research_DrillingClaws_quick", [[39]], None, [75, 75], 0, 0),
    Avail_fn(437, "Research_GlialRegeneration_quick", [[97], [100, 101]], None, [100, 100], 0, 0),
    Avail_fn(438, "Research_GroovedSpines_quick", [[91]], None, [100, 100], 0, 0),
    Avail_fn(439, "Research_HiSecAutoTracking_quick", [[22]], None, [100, 100], 0, 0),
    Avail_fn(440, "Research_HighCapacityFuelTanks_quick", [[30, 41]], None, [100, 100], 0, 0),
    Avail_fn(441, "Research_InfernalPreigniter_quick", [[39]], None, [100, 100], 0, 0),
    Avail_fn(442, "Research_MuscularAugments_quick", [[91]], None, [100, 100], 0, 0),
    Avail_fn(443, "Research_NeosteelFrame_quick", [[22]], None, [100, 100], 0, 0),
    Avail_fn(444, "Research_NeuralParasite_quick", [[94]], None, [150, 150], 0, 0),
    Avail_fn(445, "Research_PathogenGlands_quick", [[94]], None, [150, 150], 0, 0),
    Avail_fn(446, "Research_PersonalCloaking_quick", [[26]], None, [150, 150], 0, 0),
    Avail_fn(447, "Research_PneumatizedCarapace_quick", [[86, 100, 101]], None, [100, 100], 0, 0),
    Avail_fn(448, "Research_RavenCorvidReactor_quick", [[41]], None, [150, 150], 0, 0),
    Avail_fn(449, "Research_RavenRecalibratedExplosives_quick", [[41]], None, [150, 150], 0, 0),
    Avail_fn(450, "Research_SmartServos_quick", [[39]], None, [100, 100], 0, 0),
    Avail_fn(451, "Research_Stimpack_quick", [[37]], None, [100, 100], 0, 0),
    Avail_fn(452, "Research_TerranInfantryArmor_quick", [[22]], None, [100, 100], 0, 0),
    Avail_fn(456, "Research_TerranInfantryWeapons_quick", [[22]], None, [100, 100], 0, 0),
    Avail_fn(460, "Research_TerranShipWeapons_quick", [[29]], None, [100, 100], 0, 0),
    Avail_fn(464, "Research_TerranStructureArmorUpgrade_quick", [[22]], None, [150, 150], 0, 0),
    Avail_fn(465, "Research_TerranVehicleAndShipPlating_quick", [[29]], None, [100, 100], 0, 0),
    Avail_fn(469, "Research_TerranVehicleWeapons_quick", [[29]], None, [100, 100], 0, 0),
    Avail_fn(473, "Research_TunnelingClaws_quick", [[100, 101], [97]], None, [100, 100], 0, 0),
    Avail_fn(474, "Research_ZergFlyerArmor_quick", [[92, 102]], None, [150, 150], 0, 0),
    Avail_fn(478, "Research_ZergFlyerAttack_quick", [[92, 102]], None, [100, 100], 0, 0),
    Avail_fn(482, "Research_ZergGroundArmor_quick", [[90]], None, [150, 150], 0, 0),
    Avail_fn(486, "Research_ZergMeleeWeapons_quick", [[90]], None, [100, 100], 0, 0),
    Avail_fn(490, "Research_ZergMissileWeapons_quick", [[90]], None, [100, 100], 0, 0),
    Avail_fn(494, "Research_ZerglingAdrenalGlands_quick", [[89], [101]], None, [200, 200], 0, 0),
    Avail_fn(495, "Research_ZerglingMetabolicBoost_quick", [[89]], None, [100, 100], 0, 0),
    Avail_fn(498, "Train_Baneling_quick", [[105], [96]], None, [25, 25], 0, 0),
    Avail_fn(499, "Train_Banshee_quick", [[41]], None, [150, 100], 0, 0),
    Avail_fn(500, "Train_Battlecruiser_quick", [[41], [30]], None, [500, 300], 0, 0),
    Avail_fn(501, "Train_Corruptor_quick", [[151], [92, 102]], None, [150, 100], 0, 2),
    Avail_fn(502, "Train_Cyclone_quick", [[27]], None, [150, 100], 0, 0),
    Avail_fn(503, "Train_Drone_quick", [[151]], None, [50, 0], 0, 1),
    Avail_fn(504, "Train_Ghost_quick", [[37], [26]], None, [150, 125], 0, 0),
    Avail_fn(505, "Train_Hellbat_quick", [[27]], None, [100, 0], 0, 0),
    Avail_fn(506, "Train_Hellion_quick", [[27]], None, [100, 0], 0, 0),
    Avail_fn(507, "Train_Hydralisk_quick", [[151], [91]], None, [100, 50], 0, 2),
    Avail_fn(508, "Train_Infestor_quick", [[151], [94]], None, [100, 150], 0, 2),
    Avail_fn(509, "Train_Liberator_quick", [[28]], None, [150, 150], 0, 0),
    Avail_fn(510, "Train_Marauder_quick", [[37]], None, [100, 25], 0, 0),
    Avail_fn(511, "Train_Marine_quick", [[21]], None, [50, 0], 0, 0),
    Avail_fn(512, "Train_Medivac_quick", [[28]], None, [100, 100], 0, 0),
    Avail_fn(513, "Train_MothershipCore_quick", [[59]], None, [], 0, 0),
    Avail_fn(514, "Train_Mutalisk_quick", [[151], [92, 102]], None, [100, 100], 0, 2),
    Avail_fn(515, "Train_Overlord_quick", [[151]], None, [100, 0], 0, 0),
    Avail_fn(516, "Train_Queen_quick", [[86, 100, 101], [89]], None, [150, 0], 0, 0),
    Avail_fn(517, "Train_Raven_quick", [[41]], None, [100, 200], 0, 0),
    Avail_fn(518, "Train_Reaper_quick", [[21]], None, [50, 50], 0, 0),
    Avail_fn(519, "Train_Roach_quick", [[151], [97]], None, [75, 25], 0, 2),
    Avail_fn(520, "Train_SCV_quick", [[18, 132, 130]], None, [50, 0], 0, 0),
    Avail_fn(521, "Train_SiegeTank_quick", [[39]], None, [150, 125], 0, 0),
    Avail_fn(522, "Train_SwarmHost_quick", [[151], [94]], None, [100, 75], 0, 3),
    Avail_fn(523, "Train_Thor_quick", [[39]], None, [300, 200], 0, 0),
    Avail_fn(524, "Train_Ultralisk_quick", [[151], [93]], None, [300, 200], 0, 6),
    Avail_fn(525, "Train_VikingFighter_quick", [[28]], None, [150, 75], 0, 0),
    Avail_fn(526, "Train_Viper_quick", [[151], [101]], None, [100, 200], 0, 3),
    Avail_fn(527, "Train_WidowMine_quick", [[27]], None, [75, 25], 0, 0),
    Avail_fn(528, "Train_Zergling_quick", [[151], [89]], None, [50, 0], 0, 1),
    Avail_fn(537, "Effect_YamatoGun_unit", [[57]], None, [], 0, 0),
    Avail_fn(538, "Effect_KD8Charge_unit", [[49]], None, [], 0, 0),
    Avail_fn(541, "Effect_LockOn_autocast", [[692]], None, [], 0, 0),
    Avail_fn(544, "Morph_WarpGate_autocast", [[62], [84]], None, [], 0, 0),
    Avail_fn(553, "Research_AnabolicSynthesis_quick", [[93]], None, [150, 150], 0, 0),
    Avail_fn(554, "Research_CycloneLockOnDamage_quick", [[39]], None, [], 0, 0),
    Avail_fn(556, "UnloadUnit_quick", [[81, 136, 24, 18, 54, 142, 95, 893]], None, [], 0, 0),
    Avail_fn(563, "Research_EnhancedShockwaves_quick", [[26]], None, [150, 150], 0, 0),
]


def get_available_actions_raw_data(obs):
    units = obs["raw_units"]
    units_set = set()
    units_mana = collections.defaultdict(list)
    for unit in units:
        if unit.alliance == 1 and unit.build_progress >= 1:
            units_set.add(unit.unit_type)
            if unit.energy > 0:
                units_mana[unit.unit_type].append(unit.energy)
    upgrades = obs["upgrades"]
    upgrades_set = set()
    for upgrade in upgrades:
        upgrades_set.add(upgrade)
    vector = []
    for function in FUNCTION_LIST:
        if function.func_id == 111 or function.func_id == 112:
            if ((87 in upgrades_set and 74 in units_set) or (141 in upgrades_set and 76 in units_set)):
                vector.append(function)
            else:
                pass
        elif function.func_id == 232 or function.func_id == 246:
            if ((498 in units_set or 500 in units_set) or (502 in units_set) or (64 in upgrades_set)
                    or (503 in units_set)):
                vector.append(function)
            else:
                pass
        elif function.units == 0 and function.upgrade == 0:
            pass
        else:
            if function.units is None and function.upgrade is None:
                vector.append(function)
            else:
                bool_vector = []
                for units_group in function.units:
                    flag = 0
                    for unit in units_group:
                        if unit in units_set:
                            flag = 1
                    bool_vector.append(flag)
                if 0 in bool_vector:
                    pass
                else:
                    if function.upgrade is None or function.upgrade in upgrades_set:
                        vector.append(function)
                    else:
                        pass
    stat_data = obs['player'][1:]
    remove_func = []
    for func in vector:
        if func[4]:  # resource
            if (stat_data[0] < func.resource[0]
                    or stat_data[1] < func.resource[1]):
                remove_func.append(func)
                continue
        if func[5]:  # mana
            flag = False
            for unit_type in func.units[0]:
                for unit_mana in units_mana[unit_type]:
                    if unit_mana >= func.mana:
                        flag = True
            if not flag:
                remove_func.append(func)
                continue
        if func[6]:  # supply
            if stat_data[3] - stat_data[2] < func.supply:
                remove_func.append(func)
                continue

    for item in remove_func:
        vector.remove(item)
    ava_action = torch.zeros(NUM_ACTIONS)
    for t in vector:
        ava_action[ACTIONS_REORDER[t.func_id]] = 1
    return torch.FloatTensor(ava_action)


def get_available_actions_processed_data(data, check_action=False):
    alliance_start = 4 + 259
    units_set = set()
    units_mana = collections.defaultdict(list)
    for unit_type, info in zip(data['entity_raw']['type'], data['entity_info']):
        alliance = info[alliance_start:alliance_start + 5]
        if alliance[1] == 1 and info[0] >= 1:
            units_set.add(unit_type)
            if info[3] > 0:
                units_mana[unit_type].append(info[3])
    upgrades_set = set()
    for idx in torch.nonzero(data['scalar_info']['upgrades']).squeeze(1).tolist():
        upgrades_set.add(UPGRADES_REORDER_INV[idx])
    vector = []
    for function in FUNCTION_LIST:
        if function.func_id == 111 or function.func_id == 112:
            if ((87 in upgrades_set and 74 in units_set) or (141 in upgrades_set and 76 in units_set)):
                vector.append(function)
            else:
                pass
        elif function.func_id == 232 or function.func_id == 246:
            if ((498 in units_set or 500 in units_set) or (502 in units_set) or (64 in upgrades_set)
                    or (503 in units_set)):
                vector.append(function)
            else:
                pass
        elif function.units == 0 and function.upgrade == 0:
            pass
        else:
            if function.units is None and function.upgrade is None:
                vector.append(function)
            else:
                bool_vector = []
                for units_group in function.units:
                    flag = 0
                    for unit in units_group:
                        if unit in units_set:
                            flag = 1
                    bool_vector.append(flag)
                if 0 in bool_vector:
                    pass
                else:
                    if function.upgrade is None or function.upgrade in upgrades_set:
                        vector.append(function)
                    else:
                        pass
    stat_data = data['scalar_info']['agent_statistics']
    epsilon = 1e-3

    remove_func = []
    for func in vector:
        if func[4]:  # resource
            if (stat_data[0].exp() - 1 + epsilon < func.resource[0]
                    or stat_data[1].exp() - 1 + epsilon < func.resource[1]):
                remove_func.append(func)
                continue
        if func[5]:  # mana
            flag = False
            for unit_type in func.units[0]:
                for unit_mana in units_mana[unit_type]:
                    if unit_mana + epsilon >= func.mana / 200:
                        flag = True
            if not flag:
                remove_func.append(func)
                continue
        if func[6]:  # supply
            if stat_data[3].exp() - stat_data[2].exp() + epsilon < func.supply:
                remove_func.append(func)
                continue

    for item in remove_func:
        vector.remove(item)
    ava_action = torch.zeros(NUM_ACTIONS)
    for t in vector:
        ava_action[ACTIONS_REORDER[t.func_id]] = 1
    data['scalar_info']['available_actions'] = ava_action
    if check_action:
        ava_id = [i.func_id for i in vector]
        if data['actions']['action_type'].item() not in ava_id:
            data['check_action'] = False
        else:
            data['check_action'] = True
    return data


if __name__ == '__main__':
    import pickle
    from sc2learner.utils import read_file_ceph
    from sc2learner.envs.observations.alphastar_obs_wrapper import decompress_obs
    import multiprocessing
    import queue
    f = open('/mnt/lustre/zhouhang2/data/602.zerg.128.zvz', 'r')
    q = multiprocessing.Queue()
    for i in reversed(f.readlines()):
        p = i[:-1].split(' ')[0]
        q.put(p)
    f.close()

    import os

    def check_avail(q):
        print('{}: start!'.format(os.getpid()))

        while True:
            try:
                path = q.get(block=True, timeout=1)
            except queue.Empty:
                return
            data = read_file_ceph(path + '.step', read_type='pickle')
            print('read success:', path)
            for d in data:
                d = decompress_obs(d)
                if d['actions']['action_type'].item() not in (1, 2, 3, 12, 102):
                    print(d['actions']['action_type'].item(), d['actions']['delay'].item())
                get_available_actions_processed_data(d)

    check_avail(q)
