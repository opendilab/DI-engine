from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import gym
from gym import spaces
from pysc2.lib.typeenums import UNIT_TYPEID as UNIT_TYPE

from sc2learner.envs.spaces.pysc2_raw import PySC2RawObservation
from sc2learner.envs.spaces.mask_discrete import MaskDiscrete
from sc2learner.envs.common.data_context import DataContext
from sc2learner.envs.observations.spatial_features import UnitTypeCountMapFeature
from sc2learner.envs.observations.spatial_features import AllianceCountMapFeature
from sc2learner.envs.observations.nonspatial_features import PlayerFeature
from sc2learner.envs.observations.nonspatial_features import ScoreFeature
from sc2learner.envs.observations.nonspatial_features import WorkerFeature
from sc2learner.envs.observations.nonspatial_features import UnitTypeCountFeature
from sc2learner.envs.observations.nonspatial_features import UnitStatCountFeature
from sc2learner.envs.observations.nonspatial_features import GameProgressFeature
from sc2learner.envs.observations.nonspatial_features import ActionSeqFeature


class ZergObservationWrapper(gym.Wrapper):
    def __init__(self, env, use_spatial_features=False, use_game_progress=True, action_seq_len=8, use_regions=False):
        super(ZergObservationWrapper, self).__init__(env)
        # TODO: multiple observation space
        #assert isinstance(env.observation_space, PySC2RawObservation)
        self._use_spatial_features = use_spatial_features
        self._use_game_progress = use_game_progress
        self._dc = DataContext()

        # nonspatial features
        self._unit_count_feature = UnitTypeCountFeature(
            type_list=[
                UNIT_TYPE.ZERG_LARVA.value,
                UNIT_TYPE.ZERG_DRONE.value,
                UNIT_TYPE.ZERG_ZERGLING.value,
                UNIT_TYPE.ZERG_BANELING.value,
                UNIT_TYPE.ZERG_ROACH.value,
                UNIT_TYPE.ZERG_ROACHBURROWED.value,
                UNIT_TYPE.ZERG_RAVAGER.value,
                UNIT_TYPE.ZERG_HYDRALISK.value,
                UNIT_TYPE.ZERG_LURKERMP.value,
                UNIT_TYPE.ZERG_LURKERMPBURROWED.value,
                # UNIT_TYPE.ZERG_VIPER.value,
                UNIT_TYPE.ZERG_MUTALISK.value,
                UNIT_TYPE.ZERG_CORRUPTOR.value,
                UNIT_TYPE.ZERG_BROODLORD.value,
                # UNIT_TYPE.ZERG_SWARMHOSTMP.value,
                UNIT_TYPE.ZERG_LOCUSTMP.value,
                # UNIT_TYPE.ZERG_INFESTOR.value,
                UNIT_TYPE.ZERG_ULTRALISK.value,
                UNIT_TYPE.ZERG_BROODLING.value,
                UNIT_TYPE.ZERG_OVERLORD.value,
                UNIT_TYPE.ZERG_OVERSEER.value,
                # UNIT_TYPE.ZERG_CHANGELING.value,
                UNIT_TYPE.ZERG_QUEEN.value
            ],
            use_regions=use_regions
        )
        self._building_count_feature = UnitTypeCountFeature(
            type_list=[
                UNIT_TYPE.ZERG_SPINECRAWLER.value,
                UNIT_TYPE.ZERG_SPORECRAWLER.value,
                # UNIT_TYPE.ZERG_NYDUSCANAL.value,
                UNIT_TYPE.ZERG_EXTRACTOR.value,
                UNIT_TYPE.ZERG_SPAWNINGPOOL.value,
                UNIT_TYPE.ZERG_ROACHWARREN.value,
                UNIT_TYPE.ZERG_HYDRALISKDEN.value,
                UNIT_TYPE.ZERG_HATCHERY.value,
                UNIT_TYPE.ZERG_EVOLUTIONCHAMBER.value,
                UNIT_TYPE.ZERG_BANELINGNEST.value,
                UNIT_TYPE.ZERG_INFESTATIONPIT.value,
                UNIT_TYPE.ZERG_SPIRE.value,
                UNIT_TYPE.ZERG_ULTRALISKCAVERN.value,
                # UNIT_TYPE.ZERG_NYDUSNETWORK.value,
                UNIT_TYPE.ZERG_LURKERDENMP.value,
                UNIT_TYPE.ZERG_LAIR.value,
                UNIT_TYPE.ZERG_HIVE.value,
                UNIT_TYPE.ZERG_GREATERSPIRE.value
            ],
            use_regions=False
        )
        self._unit_stat_count_feature = UnitStatCountFeature(use_regions=use_regions)
        self._player_feature = PlayerFeature()
        self._score_feature = ScoreFeature()
        self._worker_feature = WorkerFeature()
        if use_game_progress:
            self._game_progress_feature = GameProgressFeature()
        self._action_seq_feature = ActionSeqFeature(self.action_space.n, action_seq_len)
        n_dims = sum(
            [
                self._unit_stat_count_feature.num_dims, self._unit_count_feature.num_dims,
                self._building_count_feature.num_dims, self._player_feature.num_dims, self._score_feature.num_dims,
                self._worker_feature.num_dims, self._action_seq_feature.num_dims,
                self._game_progress_feature.num_dims if use_game_progress else 0,
                self.env.action_space.n if isinstance(self.env.action_space, MaskDiscrete) else 0
            ]
        )

        # spatial features
        if use_spatial_features:
            resolution = self.env.observation_space.space_attr["minimap"][1]
            self._unit_type_count_map_feature = UnitTypeCountMapFeature(
                type_map={
                    UNIT_TYPE.ZERG_DRONE.value: 0,
                    UNIT_TYPE.ZERG_ZERGLING.value: 1,
                    UNIT_TYPE.ZERG_ROACH.value: 2,
                    UNIT_TYPE.ZERG_ROACHBURROWED.value: 2,
                    UNIT_TYPE.ZERG_HYDRALISK.value: 3,
                    UNIT_TYPE.ZERG_OVERLORD.value: 4,
                    UNIT_TYPE.ZERG_OVERSEER.value: 4,
                    UNIT_TYPE.ZERG_HATCHERY.value: 5,
                    UNIT_TYPE.ZERG_LAIR.value: 5,
                    UNIT_TYPE.ZERG_HIVE.value: 5,
                    UNIT_TYPE.ZERG_EXTRACTOR.value: 6,
                    UNIT_TYPE.ZERG_QUEEN.value: 7,
                    UNIT_TYPE.ZERG_RAVAGER.value: 8,
                    UNIT_TYPE.ZERG_BANELING.value: 9,
                    UNIT_TYPE.ZERG_LURKERMP.value: 10,
                    UNIT_TYPE.ZERG_LURKERMPBURROWED.value: 10,
                    UNIT_TYPE.ZERG_VIPER.value: 11,
                    UNIT_TYPE.ZERG_MUTALISK.value: 12,
                    UNIT_TYPE.ZERG_CORRUPTOR.value: 13,
                    UNIT_TYPE.ZERG_BROODLORD.value: 14,
                    UNIT_TYPE.ZERG_SWARMHOSTMP.value: 15,
                    UNIT_TYPE.ZERG_INFESTOR.value: 16,
                    UNIT_TYPE.ZERG_ULTRALISK.value: 17,
                    UNIT_TYPE.ZERG_CHANGELING.value: 18,
                    UNIT_TYPE.ZERG_SPINECRAWLER.value: 19,
                    UNIT_TYPE.ZERG_SPORECRAWLER.value: 20
                },
                resolution=resolution,
            )
            self._alliance_count_map_feature = AllianceCountMapFeature(resolution)
            n_channels = sum(
                [self._unit_type_count_map_feature.num_channels, self._alliance_count_map_feature.num_channels]
            )

        if use_spatial_features:
            if isinstance(self.env.action_space, MaskDiscrete):
                self.observation_space = spaces.Tuple(
                    [
                        spaces.Box(0.0, float('inf'), [n_channels, resolution, resolution], dtype=np.float32),
                        spaces.Box(0.0, float('inf'), [n_dims], dtype=np.float32),
                        spaces.Box(0.0, 1.0, [self.env.action_space.n], dtype=np.float32)
                    ]
                )
            else:
                self.observation_space = spaces.Tuple(
                    [
                        spaces.Box(0.0, float('inf'), [n_channels, resolution, resolution], dtype=np.float32),
                        spaces.Box(0.0, float('inf'), [n_dims], dtype=np.float32)
                    ]
                )
        else:
            if isinstance(self.env.action_space, MaskDiscrete):
                self.observation_space = spaces.Tuple(
                    [
                        spaces.Box(0.0, float('inf'), [n_dims], dtype=np.float32),
                        spaces.Box(0.0, 1.0, [self.env.action_space.n], dtype=np.float32)
                    ]
                )
            else:
                self.observation_space = spaces.Box(0.0, float('inf'), [n_dims], dtype=np.float32)

    def step(self, action):
        self._action_seq_feature.push_action(action)
        observation, reward, done, info = self.env.step(action)
        self._dc.update(observation)
        return self._observation(observation), reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset()
        self._dc.reset(observation)
        self._action_seq_feature.reset()
        return self._observation(observation)

    @property
    def action_names(self):
        if not hasattr(self.env, 'action_names'):
            raise NotImplementedError
        return self.env.action_names

    @property
    def player_position(self):
        if not hasattr(self.env, 'player_position'):
            raise NotImplementedError
        return self.env.player_position

    def _observation(self, observation):
        need_flip = True if self.env.player_position == 0 else False

        # nonspatial features
        unit_type_feat = self._unit_count_feature.features(observation, need_flip)
        building_type_feat = self._building_count_feature.features(observation, need_flip)
        unit_stat_feat = self._unit_stat_count_feature.features(observation, need_flip)
        player_feat = self._player_feature.features(observation)
        score_feat = self._score_feature.features(observation)
        worker_feat = self._worker_feature.features(self._dc)
        if self._use_game_progress:
            game_progress_feat = self._game_progress_feature.features(observation)
        action_seq_feat = self._action_seq_feature.features()
        nonspatial_feat = np.concatenate(
            [
                unit_type_feat, building_type_feat, unit_stat_feat, player_feat, score_feat, worker_feat,
                action_seq_feat, game_progress_feat if self._use_game_progress else np.array([], dtype=np.float32),
                np.array(observation['action_mask'], dtype=np.float32)
                if isinstance(self.env.action_space, MaskDiscrete) else np.array([], dtype=np.float32)
            ]
        )

        # spatial features
        if self._use_spatial_features:
            ally_map_feat = self._alliance_count_map_feature.features(observation, need_flip)
            type_map_feat = self._unit_type_count_map_feature.features(observation, need_flip)
            spatial_feat = np.concatenate([ally_map_feat, type_map_feat])

        # return features
        if self._use_spatial_features:
            if isinstance(self.env.action_space, MaskDiscrete):
                return (spatial_feat, nonspatial_feat, observation['action_mask'])
            else:
                return (spatial_feat, nonspatial_feat)
        else:
            if isinstance(self.env.action_space, MaskDiscrete):
                return (nonspatial_feat, observation['action_mask'])
            else:
                return nonspatial_feat


class ZergPlayerObservationWrapper(ZergObservationWrapper):
    def __init__(self, player, **kwargs):
        self._warn_double_wrap = lambda *args: None
        self._player = player
        super(ZergPlayerObservationWrapper, self).__init__(**kwargs)

    def step(self, action):
        self._action_seq_feature.push_action(action[self._player])
        observation, reward, done, info = self.env.step(action)
        self._dc.update(observation[self._player])
        observation[self._player] = self._observation(observation[self._player])
        return observation, reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset()
        self._dc.reset(observation[self._player])
        self._action_seq_feature.reset()
        observation[self._player] = self._observation(observation[self._player])
        return observation
