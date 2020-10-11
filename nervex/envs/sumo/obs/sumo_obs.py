from functools import reduce

import numpy as np
import torch
import traci

from nervex.envs.common import EnvElement


class SumoObs(EnvElement):
    r"""
    Overview:
        the observation element of Sumo enviroment

    Interface:
        _init, to_agent_processor
    """
    _name = "SumoObs"

    def _init(self, cfg) -> None:
        r"""
        Overview:
            init the sumo observation environment with the given config file
        Arguments:
            - cfg(:obj:`EasyDict`): config, you can refer to `envs/sumo/sumo_env_default_config.yaml`
        """
        self._cfg = cfg
        self._tls_num = len(cfg.tls)
        self._tls = cfg.tls
        self._incoming_roads = reduce(lambda x, y: x + y, cfg.incoming_roads.values())
        self._incoming_roads2tls = {t: k for k, v in cfg.incoming_roads.items() for t in v}
        # lanes
        self._incoming_roads_lanes = {k: [] for k in self._tls}
        self._tls_lanes_num = {k: 0 for k in self._tls}
        for k, v in cfg.incoming_roads_lanes.items():
            tls = self._incoming_roads2tls[k]
            self._tls_lanes_num[tls] += len(v)
            self._incoming_roads_lanes[tls] += v
        self._incoming_roads_lanes2tls = {t: k for t in v for k, v in self._incoming_roads_lanes.items()}
        self._lane_obs_num = 10
        self._tls_obs_num = {k: v * 10 for k, v in self._tls_lanes_num.items()}
        self._use_centralized_obs = cfg.use_centralized_obs
        if self._use_centralized_obs:
            self._shape = (sum([v * 10 for k, v in self._tls_lanes_num.items()]),)
        else:
            self._shape = (self._tls_obs_num,)
        self._value = {'min': 0, 'max': 1, 'dtype': float, 'dinfo': '0 or 1'}
        self._from_agent_processor = None

    def _to_agent_processor(self) -> dict:
        """
        Overview:
            return the formated observation
        Returns:
            - obs(:obj:`torch.Tensor` or :obj:`dict`): the returned observation,\
            :obj:`torch.Tensor` if used centerlized_obs, else :obj:`dict` with format {traffic_light: reward}

        """
        self._lane_lens = {t: traci.lane.getLength(t) for v in self._incoming_roads_lanes.values() for t in v}
        obs = {k: torch.zeros(v) for k, v in self._tls_obs_num.items()}
        for car_id in traci.vehicle.getIDList():
            lane_id = traci.vehicle.getLaneID(car_id)
            for tls, lanes in self._incoming_roads_lanes.items():
                if lane_id in lanes:
                    lane_pos = traci.vehicle.getLanePosition(car_id)
                    lane_idx = lanes.index(lane_id)
                    lane_len = self._lane_lens[lane_id]
                    tl_dist = lane_len - lane_pos  # the distance to traffic light
                    pos_idx = int((tl_dist / lane_len) * 10)
                    pos_idx = np.clip(pos_idx, 0, 9)
                    obs[tls][lane_idx * self._lane_obs_num + pos_idx] = 1

        if self._use_centralized_obs:
            obs = torch.cat(list(obs.values()), dim=0)
        return obs

    # override
    def _details(self):
        return '{}'.format(self._shape)
