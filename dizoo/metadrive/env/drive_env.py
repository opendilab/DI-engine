import copy
import gym
import numpy as np
from ditk import logging
from typing import Union, Dict, AnyStr, Tuple, Optional
from gym.envs.registration import register
from metadrive.manager.traffic_manager import TrafficMode
from metadrive.obs.top_down_obs_multi_channel import TopDownMultiChannel
from metadrive.constants import RENDER_MODE_NONE, DEFAULT_AGENT, REPLAY_DONE, TerminationState
from metadrive.envs.base_env import BaseEnv
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import parse_map_config, MapGenerateMethod
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.vehicle.base_vehicle import BaseVehicle
from metadrive.utils import Config, merge_dicts, get_np_random, clip
from metadrive.envs.base_env import BASE_DEFAULT_CONFIG
from metadrive.component.road_network import Road
from metadrive.component.algorithm.blocks_prob_dist import PGBlockDistConfig

METADRIVE_DEFAULT_CONFIG = dict(
    # ===== Generalization =====
    start_seed=0,
    environment_num=10,
    decision_repeat=20,
    block_dist_config=PGBlockDistConfig,

    # ===== Map Config =====
    map=3,  # int or string: an easy way to fill map_config
    random_lane_width=False,
    random_lane_num=False,
    map_config={
        BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM,
        BaseMap.GENERATE_CONFIG: None,  # it can be a file path / block num / block ID sequence
        BaseMap.LANE_WIDTH: 3.5,
        BaseMap.LANE_NUM: 3,
        "exit_length": 50,
    },

    # ===== Traffic =====
    traffic_density=0.1,
    need_inverse_traffic=False,
    traffic_mode=TrafficMode.Trigger,  # "Respawn", "Trigger"
    random_traffic=False,  # Traffic is randomized at default.
    traffic_vehicle_config=dict(
        show_navi_mark=False,
        show_dest_mark=False,
        enable_reverse=False,
        show_lidar=False,
        show_lane_line_detector=False,
        show_side_detector=False,
    ),

    # ===== Object =====
    accident_prob=0.,  # accident may happen on each block with this probability, except multi-exits block

    # ===== Others =====
    use_AI_protector=False,
    save_level=0.5,
    is_multi_agent=False,
    vehicle_config=dict(spawn_lane_index=(FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, 0)),

    # ===== Agent =====
    random_spawn_lane_index=True,
    target_vehicle_configs={
        DEFAULT_AGENT: dict(
            use_special_color=True,
            spawn_lane_index=(FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, 0),
        )
    },

    # ===== Reward Scheme =====
    # See: https://github.com/decisionforce/metadrive/issues/283
    success_reward=10.0,
    out_of_road_penalty=5.0,
    crash_vehicle_penalty=5.0,
    crash_object_penalty=5.0,
    driving_reward=1.0,
    speed_reward=0.1,
    use_lateral_reward=False,

    # ===== Cost Scheme =====
    crash_vehicle_cost=1.0,
    crash_object_cost=1.0,
    out_of_road_cost=1.0,

    # ===== Termination Scheme =====
    out_of_route_done=False,
    on_screen=False,
    show_bird_view=False,
)


class MetaDrivePPOOriginEnv(BaseEnv):

    @classmethod
    def default_config(cls) -> "Config":
        config = super(MetaDrivePPOOriginEnv, cls).default_config()
        config.update(METADRIVE_DEFAULT_CONFIG)
        config.register_type("map", str, int)
        config["map_config"].register_type("config", None)
        return config

    def __init__(self, config: dict = None):
        self.raw_cfg = config
        self.default_config_copy = Config(self.default_config(), unchangeable=True)
        self.init_flag = False

    @property
    def observation_space(self):
        return gym.spaces.Box(0, 1, shape=(84, 84, 5), dtype=np.float32)

    @property
    def action_space(self):
        return gym.spaces.Box(-1, 1, shape=(2, ), dtype=np.float32)

    @property
    def reward_space(self):
        return gym.spaces.Box(-100, 100, shape=(1, ), dtype=np.float32)

    def seed(self, seed, dynamic_seed=False):
        # TODO implement dynamic_seed mechanism
        super().seed(seed)

    def reset(self):
        if not self.init_flag:
            super(MetaDrivePPOOriginEnv, self).__init__(self.raw_cfg)
            self.start_seed = self.config["start_seed"]
            self.env_num = self.config["environment_num"]
            self.init_flag = True
        obs = super().reset()
        return obs

    def _merge_extra_config(self, config: Union[dict, "Config"]) -> "Config":
        config = self.default_config().update(config, allow_add_new_key=False)
        if config["vehicle_config"]["lidar"]["distance"] > 50:
            config["max_distance"] = config["vehicle_config"]["lidar"]["distance"]
        return config

    def _post_process_config(self, config):
        config = super(MetaDrivePPOOriginEnv, self)._post_process_config(config)
        if not config["rgb_clip"]:
            logging.warning(
                "You have set rgb_clip = False, which means the observation will be uint8 values in [0, 255]. "
                "Please make sure you have parsed them later before feeding them to network!"
            )
        config["map_config"] = parse_map_config(
            easy_map_config=config["map"], new_map_config=config["map_config"], default_config=self.default_config_copy
        )
        config["vehicle_config"]["rgb_clip"] = config["rgb_clip"]
        config["vehicle_config"]["random_agent_model"] = config["random_agent_model"]
        if config.get("gaussian_noise", 0) > 0:
            assert config["vehicle_config"]["lidar"]["gaussian_noise"] == 0, "You already provide config!"
            assert config["vehicle_config"]["side_detector"]["gaussian_noise"] == 0, "You already provide config!"
            assert config["vehicle_config"]["lane_line_detector"]["gaussian_noise"] == 0, "You already provide config!"
            config["vehicle_config"]["lidar"]["gaussian_noise"] = config["gaussian_noise"]
            config["vehicle_config"]["side_detector"]["gaussian_noise"] = config["gaussian_noise"]
            config["vehicle_config"]["lane_line_detector"]["gaussian_noise"] = config["gaussian_noise"]
        if config.get("dropout_prob", 0) > 0:
            assert config["vehicle_config"]["lidar"]["dropout_prob"] == 0, "You already provide config!"
            assert config["vehicle_config"]["side_detector"]["dropout_prob"] == 0, "You already provide config!"
            assert config["vehicle_config"]["lane_line_detector"]["dropout_prob"] == 0, "You already provide config!"
            config["vehicle_config"]["lidar"]["dropout_prob"] = config["dropout_prob"]
            config["vehicle_config"]["side_detector"]["dropout_prob"] = config["dropout_prob"]
            config["vehicle_config"]["lane_line_detector"]["dropout_prob"] = config["dropout_prob"]
        target_v_config = copy.deepcopy(config["vehicle_config"])
        if not config["is_multi_agent"]:
            target_v_config.update(config["target_vehicle_configs"][DEFAULT_AGENT])
            config["target_vehicle_configs"][DEFAULT_AGENT] = target_v_config
        return config

    def step(self, actions: Union[np.ndarray, Dict[AnyStr, np.ndarray]]):
        actions = self._preprocess_actions(actions)
        engine_info = self._step_simulator(actions)
        o, r, d, i = self._get_step_return(actions, engine_info=engine_info)
        return o, r, d, i

    def cost_function(self, vehicle_id: str):
        vehicle = self.vehicles[vehicle_id]
        step_info = dict()
        step_info["cost"] = 0
        if self._is_out_of_road(vehicle):
            step_info["cost"] = self.config["out_of_road_cost"]
        elif vehicle.crash_vehicle:
            step_info["cost"] = self.config["crash_vehicle_cost"]
        elif vehicle.crash_object:
            step_info["cost"] = self.config["crash_object_cost"]
        return step_info['cost'], step_info

    def _is_out_of_road(self, vehicle):
        ret = vehicle.on_yellow_continuous_line or vehicle.on_white_continuous_line or \
              (not vehicle.on_lane) or vehicle.crash_sidewalk
        if self.config["out_of_route_done"]:
            ret = ret or vehicle.out_of_route
        return ret

    def done_function(self, vehicle_id: str):
        vehicle = self.vehicles[vehicle_id]
        done = False
        done_info = {
            TerminationState.CRASH_VEHICLE: False,
            TerminationState.CRASH_OBJECT: False,
            TerminationState.CRASH_BUILDING: False,
            TerminationState.OUT_OF_ROAD: False,
            TerminationState.SUCCESS: False,
            TerminationState.MAX_STEP: False,
            TerminationState.ENV_SEED: self.current_seed,
        }
        if self._is_arrive_destination(vehicle):
            done = True
            logging.info("Episode ended! Reason: arrive_dest.")
            done_info[TerminationState.SUCCESS] = True
        if self._is_out_of_road(vehicle):
            done = True
            logging.info("Episode ended! Reason: out_of_road.")
            done_info[TerminationState.OUT_OF_ROAD] = True
        if vehicle.crash_vehicle:
            done = True
            logging.info("Episode ended! Reason: crash vehicle ")
            done_info[TerminationState.CRASH_VEHICLE] = True
        if vehicle.crash_object:
            done = True
            done_info[TerminationState.CRASH_OBJECT] = True
            logging.info("Episode ended! Reason: crash object ")
        if vehicle.crash_building:
            done = True
            done_info[TerminationState.CRASH_BUILDING] = True
            logging.info("Episode ended! Reason: crash building ")
        if self.config["max_step_per_agent"] is not None and \
                self.episode_lengths[vehicle_id] >= self.config["max_step_per_agent"]:
            done = True
            done_info[TerminationState.MAX_STEP] = True
            logging.info("Episode ended! Reason: max step ")

        if self.config["horizon"] is not None and \
                self.episode_lengths[vehicle_id] >= self.config["horizon"] and not self.is_multi_agent:
            # single agent horizon has the same meaning as max_step_per_agent
            done = True
            done_info[TerminationState.MAX_STEP] = True
            logging.info("Episode ended! Reason: max step ")

        done_info[TerminationState.CRASH] = (
            done_info[TerminationState.CRASH_VEHICLE] or done_info[TerminationState.CRASH_OBJECT]
            or done_info[TerminationState.CRASH_BUILDING]
        )
        return done, done_info

    def reward_function(self, vehicle_id: str):
        """
        Override this func to get a new reward function
        :param vehicle_id: id of BaseVehicle
        :return: reward
        """
        vehicle = self.vehicles[vehicle_id]
        step_info = dict()

        # Reward for moving forward in current lane
        if vehicle.lane in vehicle.navigation.current_ref_lanes:
            current_lane = vehicle.lane
            positive_road = 1
        else:
            current_lane = vehicle.navigation.current_ref_lanes[0]
            current_road = vehicle.navigation.current_road
            positive_road = 1 if not current_road.is_negative_road() else -1
        long_last, _ = current_lane.local_coordinates(vehicle.last_position)
        long_now, lateral_now = current_lane.local_coordinates(vehicle.position)

        # reward for lane keeping, without it vehicle can learn to overtake but fail to keep in lane
        if self.config["use_lateral_reward"]:
            lateral_factor = clip(1 - 2 * abs(lateral_now) / vehicle.navigation.get_current_lane_width(), 0.0, 1.0)
        else:
            lateral_factor = 1.0

        reward = 0.0
        reward += self.config["driving_reward"] * (long_now - long_last) * lateral_factor * positive_road
        reward += self.config["speed_reward"] * (vehicle.speed / vehicle.max_speed) * positive_road

        step_info["step_reward"] = reward

        if self._is_arrive_destination(vehicle):
            reward = +self.config["success_reward"]
        elif self._is_out_of_road(vehicle):
            reward = -self.config["out_of_road_penalty"]
        elif vehicle.crash_vehicle:
            reward = -self.config["crash_vehicle_penalty"]
        elif vehicle.crash_object:
            reward = -self.config["crash_object_penalty"]
        return reward, step_info

    def _get_reset_return(self):
        ret = {}
        self.engine.after_step()
        for v_id, v in self.vehicles.items():
            self.observations[v_id].reset(self, v)
            ret[v_id] = self.observations[v_id].observe(v)
        return ret if self.is_multi_agent else self._wrap_as_single_agent(ret)

    def switch_to_third_person_view(self) -> (str, BaseVehicle):
        if self.main_camera is None:
            return
        self.main_camera.reset()
        if self.config["prefer_track_agent"] is not None and self.config["prefer_track_agent"] in self.vehicles.keys():
            new_v = self.vehicles[self.config["prefer_track_agent"]]
            current_track_vehicle = new_v
        else:
            if self.main_camera.is_bird_view_camera():
                current_track_vehicle = self.current_track_vehicle
            else:
                vehicles = list(self.engine.agents.values())
                if len(vehicles) <= 1:
                    return
                if self.current_track_vehicle in vehicles:
                    vehicles.remove(self.current_track_vehicle)
                new_v = get_np_random().choice(vehicles)
                current_track_vehicle = new_v
        self.main_camera.track(current_track_vehicle)
        return

    def switch_to_top_down_view(self):
        self.main_camera.stop_track()

    def setup_engine(self):
        super(MetaDrivePPOOriginEnv, self).setup_engine()
        self.engine.accept("b", self.switch_to_top_down_view)
        self.engine.accept("q", self.switch_to_third_person_view)
        from metadrive.manager.traffic_manager import TrafficManager
        from metadrive.manager.map_manager import MapManager
        self.engine.register_manager("map_manager", MapManager())
        self.engine.register_manager("traffic_manager", TrafficManager())

    def _is_arrive_destination(self, vehicle):
        long, lat = vehicle.navigation.final_lane.local_coordinates(vehicle.position)
        flag = (vehicle.navigation.final_lane.length - 5 < long < vehicle.navigation.final_lane.length + 5) and (
            vehicle.navigation.get_current_lane_width() / 2 >= lat >=
            (0.5 - vehicle.navigation.get_current_lane_num()) * vehicle.navigation.get_current_lane_width()
        )
        return flag

    def _reset_global_seed(self, force_seed=None):
        """
        Current seed is set to force seed if force_seed is not None.
        Otherwise, current seed is randomly generated.
        """
        current_seed = force_seed if force_seed is not None else \
            get_np_random(self._DEBUG_RANDOM_SEED).randint(self.start_seed, self.start_seed + self.env_num)
        self.seed(current_seed)

    def _get_observations(self):
        return {DEFAULT_AGENT: self.get_single_observation(self.config["vehicle_config"])}

    def get_single_observation(self, _=None):
        return TopDownMultiChannel(
            self.config["vehicle_config"],
            self.config["on_screen"],
            self.config["rgb_clip"],
            frame_stack=3,
            post_stack=10,
            frame_skip=1,
            resolution=(84, 84),
            max_distance=36,
        )

    def clone(self, caller: str):
        cfg = copy.deepcopy(self.raw_cfg)
        return MetaDrivePPOOriginEnv(cfg)
