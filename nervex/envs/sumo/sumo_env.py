import copy
import os
from collections import namedtuple
import sys
from sumolib import checkBinary
from nervex.envs.env.base_env import BaseEnv
from nervex.envs.sumo.action.sumo_action_runner import SumoRawAction, SumoRawActionRunner
from nervex.envs.sumo.reward.sumo_reward_runner import SumoReward, SumoRewardRunner
from nervex.envs.sumo.obs.sumo_obs_runner import SumoObs, SumoObsRunner
import numpy as np

import time
import traci
from functools import reduce


class SumoEnv(BaseEnv):
    r"""

    """
    timestep = namedtuple('SumoTimestep', ['obs', 'reward', 'done', 'info'])
    # info_template = namedtuple('BaseEnvInfo', ['obs_space', 'act_space', 'rew_space', 'frame_skip', 'rep_prob'])

    def __init__(self, cfg, name, sumocfg_path, max_episode_steps, green_duration=10, yellow_duration=3, 
                 reward_type=None, inference=False):
        self._cfg = cfg
        # self._env = traci
        self.name = name
        self.max_episode_steps = max_episode_steps

        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = 80
        if not isinstance(reward_type, list):
            reward_type = [reward_type]
        assert set(reward_type).issubset(['wait_time', 'queue_len', 'delay_time']), set(reward_type)
        self.reward_type = reward_type
        self.incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        self.inference = inference

        self._sumo_cmd = self._launch_env(max_steps=max_episode_steps, sumocfg_path=sumocfg_path)

        self._action_helper = SumoRawActionRunner()
        self._reward_helper = SumoRewardRunner(self.reward_type)
        self._obs_helper = SumoObsRunner()

        self.PHASE_NS_GREEN = 0  # action 0 code 00
        self.PHASE_NSL_GREEN = 2  # action 1 code 01
        self.PHASE_EW_GREEN = 4  # action 2 code 10
        self.PHASE_EWL_GREEN = 6  # action 3 code 11

        self._launch_env_flag = True
        #WJ
        self.tls = ['htxdj_wjj', 'haxl_wjj', 'haxl_htxdj']
        self.tls_green_action = [[0, 3], [0, 3], [0, 2, 4]]
        self.tls_yellow_action = [[1, 4], [1, 4], [1, 3, 5]]
        self.action_dim = [len(t) for t in self.tls_green_action]
        self.incoming_roads = ['wjj_s3', 'htxdj_e5', 'htxdj_w4', 'wjj_n4'] + ['wjj_s5', 'haxl_e7', 'haxl_w6', 'wjj_n6']\
        + ['htxdj_e3.94', 'haxl_e6', 'haxl_w5', 'htxdj_w2.141']
        self.incoming_roads_lanes = {
            'wjj_s3': ['wjj_s3_0', 'wjj_s3_1', 'wjj_s3_2'],
            'htxdj_e5': ['htxdj_e5_0', 'htxdj_e5_1', 'htxdj_e5_2'],
            'htxdj_w4': ['htxdj_w4_0', 'htxdj_w4_1', 'htxdj_w4_2'],
            'wjj_n4': ['wjj_n4_0', 'wjj_n4_1', 'wjj_n4_2'],
            'wjj_s5': ['wjj_s5_0', 'wjj_s5_1', 'wjj_s5_2'],
            'haxl_e7': ['haxl_e7_0', 'haxl_e7_1', 'haxl_e7_2'],
            'haxl_w6': ['haxl_w6_0', 'haxl_w6_1', 'haxl_w6_2'],
            'wjj_n6': ['wjj_n6_0', 'wjj_n6_1', 'wjj_n6_2'],
            'htxdj_e3.94': ['htxdj_e3.94_0', 'htxdj_e3.94_1', 'htxdj_e3.94_2', 'htxdj_e3.94_3'],
            'haxl_e6': ['haxl_e6_0', 'haxl_e6_1', 'haxl_e6_2'],
            'haxl_w5': ['haxl_w5_0', 'haxl_w5_1', 'haxl_w5_2'],
            'htxdj_w2.141': ['htxdj_w2.141_0', 'htxdj_w2.141_1', 'htxdj_w2.141_2', 'htxdj_w2.141_3'],
        }
        self.incoming_roads_lanes_num = [len(v) for v in self.incoming_roads_lanes.values()]
        self.traffic_num = len(self.tls)
        self.state_dim = sum(self.incoming_roads_lanes_num) * 10
        self.lanes = reduce(lambda x, y: x+y, self.incoming_roads_lanes.values())

    def _launch_env(self, gui=False, sumocfg_path='sumo_config.sumocfg', max_steps=1000):
        # set gui=True can get visualization simulation result with sumo, apply gui=False in the normal training
        # and test setting

        # sumo things - we need to import python modules from the $SUMO_HOME/tools directory
        if 'SUMO_HOME' in os.environ:
            tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
            sys.path.append(tools)
        else:
            sys.exit("please declare environment variable 'SUMO_HOME'")

        # setting the cmd mode or the visual mode
        if gui is False:
            sumoBinary = checkBinary('sumo')
        else:
            sumoBinary = checkBinary('sumo-gui')

        # setting the cmd command to run sumo at simulation time
        sumo_cmd = [sumoBinary, "-c", sumocfg_path, "--no-step-log", "true", "--waiting-time-memory", str(max_steps)]

        self._launch_env_flag = True

        return sumo_cmd

    def reset(self):
        self.steps = 0
        self.last_total_wait = 0
        self.last_action = None
        self.last_vehicle_info = {}
        self._waiting_times = {}
        self.label = time.time()
        traci.start(self._sumo_cmd, label=self.label)
        if hasattr(self, 'lanes'):
            self.lane_lens = {lane_id: traci.lane.getLength(lane_id) for lane_id in self.lanes}
        # return self._get_state()
        ret = self._get_state()
        self._reward_helper.reset()
        self._obs_helper.reset()
        self._action_helper.reset()
        return ret

    def close(self):
        traci.close()

    def step(self, action: int) -> 'SumoEnv.timestep':
        assert self._launch_env_flag
        if self.steps != 0:
            self._set_yellow_phase(self.last_action)
            self._simulate(self._yellow_duration)
        self._set_green_phase(action)
        self._simulate(self._green_duration)
        self.obs = self._get_state()
        self.reward = self._reward_helper.get(self) if not self.inference else 0.
        self.done = self.steps >= self.max_episode_steps
        self.info = {}
        self.last_action = action
        # return obs, reward, done, info
        return SumoEnv.timestep(obs=self.obs, reward=self.reward, done=self.done, info=self.info)

        """
        self.agent_action = action

        #env step
        self._sumo_obs, self._reward_of_action, self._is_gameover, self._rest_life = self._env.step(action)
        self._sumo_obs = self._sumo_obs.transpose((2, 0, 1))

        #transform obs, reward and record statistics

        self.action = self._action_helper.get(self)
        self.reward = self._reward_helper.get(self)
        self.obs = self._obs_helper.get(self)

        return sumoEnv.timestep(obs=self.obs, reward=self.reward, done=self._is_gameover, rest_lives=self._rest_life)
        """

    def seed(self, seed: int) -> None:
        pass

    def _simulate(self, steps):
        for i in range(steps):
            traci.simulationStep()
            self.steps += 1

    def _set_yellow_phase(self, action):
        '''
        last_action = self.last_action
        if last_action == action:
            return
        yellow_phase_code = last_action * 2 + 1  # obtain the yellow phase code, based on the last action (ref on environment.net.xml)
        traci.trafficlight.setPhase("TL", yellow_phase_code)
        '''
        assert isinstance(action, list) or isinstance(action, tuple) or isinstance(action, np.ndarray)
        if isinstance(action, np.ndarray):
            assert len(action.shape) == 1  # one dim array
            action = action.tolist()
        # init state
        if self.last_action is None:
            return
        for idx, (act, last_act) in enumerate(zip(action, self.last_action)):
            if act != last_act:
                yellow_idx = self.tls_green_action[idx].find(last_act)
                yellow_phase_code = self.tls_yellow_action[idx][yellow_idx]
                traci.trafficlight.setPhase(self.tls[idx], yellow_phase_code)

    def _set_green_phase(self, action):
        """
        Activate the correct green light combination in sumo
        """
        '''
        if action_number == 0:
            traci.trafficlight.setPhase("TL", self.PHASE_NS_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase("TL", self.PHASE_NSL_GREEN)
        elif action_number == 2:
            traci.trafficlight.setPhase("TL", self.PHASE_EW_GREEN)
        elif action_number == 3:
            traci.trafficlight.setPhase("TL", self.PHASE_EWL_GREEN)
        else:
            raise ValueError("{}".format(action_number))
        '''
        assert isinstance(action, list) or isinstance(action, tuple) or isinstance(action, np.ndarray)
        if isinstance(action, np.ndarray):
            assert len(action.shape) == 1  # one dim array
            action = action.tolist()
        assert len(action) == len(self.tls)
        for idx, act in enumerate(action):
            traci.trafficlight.setPhase(self.tls[idx], self.tls_green_action[idx][act])

    def _get_state(self):
        '''
        # """
        # Retrieve the state of the intersection from sumo, in the form of cell occupancy
        # """
        # state = np.zeros(self._num_states)
        # car_list = traci.vehicle.getIDList()

        # for car_id in car_list:
        #     lane_pos = traci.vehicle.getLanePosition(car_id)
        #     lane_id = traci.vehicle.getLaneID(car_id)
        #     lane_pos = 750 - lane_pos
        # # inversion of lane pos, so if the car is close to the traffic light -> lane_pos = 0 --- 750 = max len of a road

        #     # distance in meters from the traffic light -> mapping into cells
        #     if lane_pos < 7:
        #         lane_cell = 0
        #     elif lane_pos < 14:
        #         lane_cell = 1
        #     elif lane_pos < 21:
        #         lane_cell = 2
        #     elif lane_pos < 28:
        #         lane_cell = 3
        #     elif lane_pos < 40:
        #         lane_cell = 4
        #     elif lane_pos < 60:
        #         lane_cell = 5
        #     elif lane_pos < 100:
        #         lane_cell = 6
        #     elif lane_pos < 160:
        #         lane_cell = 7
        #     elif lane_pos < 400:
        #         lane_cell = 8
        #     elif lane_pos <= 750:
        #         lane_cell = 9

        #     # finding the lane where the car is located
        #     # x2TL_3 are the "turn left only" lanes
        #     if lane_id == "W2TL_0" or lane_id == "W2TL_1" or lane_id == "W2TL_2":
        #         lane_group = 0
        #     elif lane_id == "W2TL_3":
        #         lane_group = 1
        #     elif lane_id == "N2TL_0" or lane_id == "N2TL_1" or lane_id == "N2TL_2":
        #         lane_group = 2
        #     elif lane_id == "N2TL_3":
        #         lane_group = 3
        #     elif lane_id == "E2TL_0" or lane_id == "E2TL_1" or lane_id == "E2TL_2":
        #         lane_group = 4
        #     elif lane_id == "E2TL_3":
        #         lane_group = 5
        #     elif lane_id == "S2TL_0" or lane_id == "S2TL_1" or lane_id == "S2TL_2":
        #         lane_group = 6
        #     elif lane_id == "S2TL_3":
        #         lane_group = 7
        #     else:
        #         lane_group = -1

        #     if lane_group >= 1 and lane_group <= 7:
        #         car_position = int(str(lane_group) + str(lane_cell))  # composition of the two postion ID to create a number in interval 0-79
        #         valid_car = True
        #     elif lane_group == 0:
        #         car_position = lane_cell
        #         valid_car = True
        #     else:
        #         valid_car = False  # flag for not detecting cars crossing the intersection or driving away from it

        #     if valid_car:
        #         state[car_position] = 1  # write the position of the car car_id in the state array in the form of "cell occupied"
        # return state
        '''
        state = np.zeros(self.state_dim)
        car_list = traci.vehicle.getIDList()

        for car_id in car_list:
            #coming_tls_id = traci.vehicle.getNextTLS(car_id)[0][0]  # the first tls, tls_id
            edge_id = traci.vehicle.getRoadID(car_id)
            if edge_id in self.incoming_roads:
                lane_pos = traci.vehicle.getLanePosition(car_id)
                lane_id = traci.vehicle.getLaneID(car_id)
                edge_idx = self.incoming_roads.index(edge_id)
                lane_idx = self.incoming_roads_lanes[edge_id].index(lane_id)
                edge_cum_idx = reduce(lambda x, y: x+y, self.incoming_roads_lanes_num[:edge_idx]) if edge_idx > 0 else 0
                lane_idx = edge_cum_idx + lane_idx

                #lane_len = traci.lane.getLength(lane_id)
                lane_len = self.lane_lens[lane_id]
                tl_dist = lane_len - lane_pos  # the distance to traffic light
                pos_idx = int((tl_dist / lane_len) * 10)
                pos_idx = np.clip(pos_idx, 0, 9)
                final_idx = lane_idx * 10 + pos_idx
                state[final_idx] = 1

        return state

    def _collect_delay_time(self):
        car_list = traci.vehicle.getIDList()
        cur_vehicle_info = {car_id: {'time': traci.vehicle.getLastActionTime(car_id), 'distance': traci.vehicle.getDistance(car_id)} for car_id in car_list}
        delay_time_sum = 0.
        valid_car_num = 0
        for car_id in cur_vehicle_info.keys():
            if car_id in self.last_vehicle_info.keys():
                real_distance = cur_vehicle_info[car_id]['distance'] - self.last_vehicle_info[car_id]['distance']
                target_speed = traci.vehicle.getMaxSpeed(car_id)
                target_distance = (cur_vehicle_info[car_id]['time'] - self.last_vehicle_info[car_id]['time']) * target_speed
                delay_time = (real_distance - target_distance) / target_speed
                delay_time_sum += delay_time
                valid_car_num += 1
        self.last_vehicle_info = cur_vehicle_info
        delay_reward = delay_time_sum / (valid_car_num + 1e-8)
        return delay_reward

    def _collect_waiting_times(self):
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
            if road_id in self.incoming_roads:  # consider only the waiting times of cars in incoming roads
                self._waiting_times[car_id] = wait_time
            else:
                if car_id in self._waiting_times:  # a car that was tracked has cleared the intersection
                    del self._waiting_times[car_id]
        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time

    def _get_queue_length(self):
        queue_length = 0
        for r in self.incoming_roads:
            queue_length += traci.edge.getLastStepHaltingNumber(r)
        return queue_length

    def info(self):
        pass

    # def info(self) -> 'sumoEnv.info':
    #     info_data = {
    #         'obs_space': self._obs_helper.info,
    #         'act_space': self._action_helper.info,
    #         'rew_space': self._reward_helper.info,
    #         'frame_skip': self.frameskip,
    #         'rep_prob': self.rep_prob
    #     }
    #     return sumoEnv.info_template(**info_data)

    def __repr__(self) -> str:
        return 'sumoEnv:\n\
                \tobservation[{}]\n\
                \taction[{}]\n\
                \treward[{}]\n'.format(repr(self._obs_helper), repr(self._action_helper), repr(self._reward_helper))
    # @property
    # def agent_action(self) -> int:
    #     return self._agent_action

    # @agent_action.setter
    # def agent_action(self, _agent_action) -> None:
    #     self._agent_action = _agent_action

    # @property
    # def reward_of_action(self) -> float:
    #     return self._reward_of_action

    # @reward_of_action.setter
    # def reward_of_action(self, _reward_of_action) -> None:
    #     self._reward_of_action = _reward_of_action

    # @property
    # def sumo_obs(self) -> np.ndarray:
    #     return self._sumo_obs

    # @sumo_obs.setter
    # def sumo_obs(self, _obs) -> None:
    #     self._sumo_obs = _obs


SumoTimestep = SumoEnv.timestep
r"""
sumo = SumoEnv({}, "sumo", "/nervex/envs/sumo/beijing_wj/sumo_config.sumocfg", max_episode_steps=2000, green_duration=10, yellow_duration=3,reward_type='delay_time')
"""
