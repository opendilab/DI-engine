from functools import reduce
from typing import Tuple, Union

import torch
import traci

from nervex.envs.common import EnvElement


class SumoReward(EnvElement):
    r"""
    Overview:
        the reward element of Sumo enviroment

    Interface:
        _init, to_agent_processor
    """
    _name = "SumoReward"

    def _init(self, cfg: dict) -> None:
        r"""
        Overview:
            init the sumo reward environment with the given config file
        Arguments:
            - cfg(:obj:`EasyDict`): config, you can refer to `envs/sumo/sumo_env_default_config.yaml`
        """
        self._cfg = cfg
        self._reduce_by_tl = cfg.reduce_by_tl
        reward_type = cfg.reward_type
        if not isinstance(reward_type, list):
            reward_type = [reward_type]
        self._reward_keys = ['wait_time', 'queue_len', 'delay_time']
        assert set(reward_type).issubset(self._reward_keys), set(reward_type)
        self._reward_type = reward_type

        self._incoming_roads = cfg.incoming_roads
        self._total_incoming_roads = reduce(lambda x, y: x + y, self._incoming_roads.values())
        self._tls = cfg.tls
        self._road2tls = {t: k for k, v in self._incoming_roads.items() for t in v}
        single_reward_shape = (1, ) if self._reduce_by_tl else {t: (1, ) for t in cfg.tls}
        self._shape = {k: single_reward_shape for k in self._reward_keys}
        self._value = {
            'wait_time': {
                'min': '-inf',
                'max': 'inf',
                'dtype': float
            },
            'queue_len': {
                'min': '-inf',
                'max': 0,
                'dtype': float
            },
            'delay_time': {
                'min': '-inf',
                'max': 'inf',
                'dtype': float
            }
        }
        self._from_agent_processor = None

    def _get_wait_time(self, data: dict) -> Tuple[Union[float, dict], dict]:
        car_list = traci.vehicle.getIDList()
        tracking_cars = [car_id for car_id in car_list if traci.vehicle.getRoadID(car_id) in self._total_incoming_roads]
        current_wait = {c: traci.vehicle.getAccumulatedWaitingTime(c) for c in tracking_cars}
        wait_time_reward = {}
        last_wait = data['last_wait_time']
        for k, v in current_wait.items():
            if k in last_wait.keys():
                wait_time_reward[k] = last_wait[k] - v
            else:
                wait_time_reward[k] = -v
        if self._reduce_by_tl:
            t = wait_time_reward.values()
            return torch.FloatTensor([sum(t) / (len(t) + 1e-8)]), current_wait
        else:
            wait_time_reward_tl = {t: 0. for t in self._tls}
            for k, v in wait_time_reward:
                tl = self._road2tls[traci.vehicle.getRoadID(k)]
                wait_time_reward_tl[tl] = (wait_time_reward_tl[tl] + v) / 2
            return torch.FloatTensor([wait_time_reward_tl]), current_wait

    def _get_queue_len(self, data: dict) -> Union[float, dict]:
        queue_len_reward = {}
        for k, v in self._incoming_roads.items():
            queue_len_reward[k] = sum([-1. * traci.edge.getLastStepHaltingNumber(r) for r in v])
        if self._reduce_by_tl:
            queue_len_reward = sum(queue_len_reward.values())
        return torch.FloatTensor([queue_len_reward])

    def _get_delay_time(self, data: dict) -> Tuple[Union[float, dict], dict]:
        car_list = traci.vehicle.getIDList()
        cur_vehicle_info = {
            car_id: {
                'time': traci.vehicle.getLastActionTime(car_id),
                'distance': traci.vehicle.getDistance(car_id)
            }
            for car_id in car_list if traci.vehicle.getRoadID(car_id) in self._total_incoming_roads
        }
        last_vehicle_info = data['last_vehicle_info']
        delay_time_reward = {}
        for car_id in cur_vehicle_info.keys():
            if car_id in last_vehicle_info.keys():
                real_distance = cur_vehicle_info[car_id]['distance'] - last_vehicle_info[car_id]['distance']
                target_speed = traci.vehicle.getMaxSpeed(car_id)
                target_distance = (cur_vehicle_info[car_id]['time'] - last_vehicle_info[car_id]['time']) * target_speed
                delay_time_reward[car_id] = (real_distance - target_distance) / (target_speed + 1e-8)
        if self._reduce_by_tl:
            t = delay_time_reward.values()
            return torch.FloatTensor([sum(t) / (len(t) + 1e-8)]), cur_vehicle_info
        else:
            delay_time_reward_tl = {t: 0. for t in self._tls}
            for k, v in delay_time_reward:
                tl = self._road2tls[traci.vehicle.getRoadID(k)]
                delay_time_reward_tl[tl] = (delay_time_reward_tl[tl] + v) / 2
            return torch.FloatTensor([delay_time_reward_tl]), cur_vehicle_info

    def _to_agent_processor(self, data: dict) -> dict:
        r"""
        Overview:
            return the raw_reward
        Returns:
            - reward(:obj:`dict` or :obj:`float`): different type reward with format {reward_type: val}
        """
        reward = {}
        assert set(data.keys()) == set(self._reward_type)
        for k, item in data.items():
            reward[k] = getattr(self, '_get_' + k)(item)
        return reward

    # override
    def _details(self):
        return 'reward_type: {}\treduce_by_tl: {}'.format(self._reward_type, self._reduce_by_tl)
