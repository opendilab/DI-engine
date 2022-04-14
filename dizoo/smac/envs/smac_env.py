import copy
import enum
from collections import namedtuple
from operator import attrgetter

import numpy as np
import math
from easydict import EasyDict
import ctools.pysc2.env.sc2_env as sc2_env
from ctools.pysc2.env.sc2_env import SC2Env
from ctools.pysc2.lib import protocol
from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import debug_pb2 as d_pb
from s2clientprotocol import sc2api_pb2 as sc_pb
from ding.envs import BaseEnv
from ding.envs.common.env_element import EnvElement, EnvElementInfo
from ding.utils import ENV_REGISTRY, deep_merge_dicts

from .smac_map import get_map_params
from .smac_action import SMACAction, distance
from .smac_reward import SMACReward

races = {
    "R": sc_common.Random,
    "P": sc_common.Protoss,
    "T": sc_common.Terran,
    "Z": sc_common.Zerg,
}

ORIGINAL_AGENT = "me"
OPPONENT_AGENT = "opponent"

SUPPORT_MAPS = [
    "SMAC_Maps_two_player/3s5z.SC2Map",
    "SMAC_Maps_two_player/3m.SC2Map",
    "GBU_Maps/infestor_viper.sc2map",
]

FORCE_RESTART_INTERVAL = 50000


class Direction(enum.IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3


@ENV_REGISTRY.register('smac')
class SMACEnv(SC2Env, BaseEnv):
    """
    This environment provides the interface for both single agent and multiple agents (two players) in
    SC2 environment.
    """

    SMACTimestep = namedtuple('SMACTimestep', ['obs', 'reward', 'done', 'info', 'episode_steps'])
    SMACEnvInfo = namedtuple('SMACEnvInfo', ['agent_num', 'obs_space', 'act_space', 'rew_space', 'episode_limit'])
    config = dict(
        two_player=False,
        mirror_opponent=False,
        reward_type="original",
        save_replay_episodes=None,
        difficulty=7,
        reward_death_value=10,
        reward_win=200,
        obs_alone=False,
        game_steps_per_episode=None,
        reward_only_positive=True,
        death_mask=False,
        special_global_state=False,
        # add map's center location ponit or not
        add_center_xy=True,
        independent_obs=False,
        # add agent's id information or not in special global state
        state_agent_id=True,
    )

    def __init__(
        self,
        cfg,
    ):
        cfg = deep_merge_dicts(EasyDict(self.config), cfg)
        self.cfg = cfg
        self.save_replay_episodes = cfg.save_replay_episodes
        assert (self.save_replay_episodes is None) or isinstance(
            self.save_replay_episodes, int
        )  # Denote the number of replays to save
        self.two_player = cfg.two_player
        self.difficulty = cfg.difficulty
        self.obs_alone = cfg.obs_alone
        self.game_steps_per_episode = cfg.game_steps_per_episode

        map_name = cfg.map_name
        assert map_name is not None
        map_params = get_map_params(map_name)
        self.reward_only_positive = cfg.reward_only_positive
        self.difficulty = cfg.difficulty
        self.obs_alone = cfg.obs_alone
        self.players, self.num_players = self._get_players(
            "agent_vs_agent" if self.two_player else "game_vs_bot",
            player1_race=map_params["a_race"],
            player2_race=map_params["b_race"]
        )
        self._map_name = map_name

        # SMAC used
        self.n_agents = map_params["n_agents"]
        self.n_enemies = map_params["n_enemies"]
        self.episode_limit = map_params["limit"]

        self._agent_race = map_params["a_race"]
        self._bot_race = map_params["b_race"]
        self.shield_bits_ally = 1 if self._agent_race == "P" else 0
        self.shield_bits_enemy = 1 if self._bot_race == "P" else 0
        self.unit_type_bits = map_params["unit_type_bits"]
        self.map_type = map_params["map_type"]

        self.agents = {}
        self.enemies = {}
        self._episode_count = 0
        self._episode_steps = 0
        self._total_steps = 0
        self._next_reset_steps = FORCE_RESTART_INTERVAL

        self._obs = None
        self.battles_won = 0
        self.battles_game = 0
        self.timeouts = 0
        self.force_restarts = 0
        self.last_stats = None

        self._min_unit_type = 0
        self.marine_id = self.marauder_id = self.medivac_id = 0
        self.hydralisk_id = self.zergling_id = self.baneling_id = 0
        self.stalker_id = self.colossus_id = self.zealot_id = 0

        self.add_center_xy = cfg.add_center_xy
        self.state_agent_id = cfg.state_agent_id
        self.death_mask = cfg.death_mask
        self.special_global_state = cfg.special_global_state

        # reward
        self.reward_death_value = cfg.reward_death_value
        self.reward_win = cfg.reward_win
        self.reward_defeat = 0
        self.reward_negative_scale = 0.5
        self.reward_type = cfg.reward_type
        self.max_reward = (self.n_enemies * self.reward_death_value + self.reward_win)
        self.obs_pathing_grid = False
        self.obs_own_health = True
        self.obs_all_health = True
        self.obs_instead_of_state = False
        self.obs_last_action = True
        self.obs_terrain_height = False
        self.obs_timestep_number = False
        self.state_last_action = True
        self.state_timestep_number = False
        if self.obs_all_health:
            self.obs_own_health = True
        self.n_obs_pathing = 8
        self.n_obs_height = 9
        self._move_amount = 2
        self.continuing_episode = False

        self._seed = None
        self._launch_env_flag = True
        self.just_force_restarts = False

        # Set to false if you need structured observation / state
        self.flatten_observation = True
        self.mirror_opponent = cfg.mirror_opponent
        if self.mirror_opponent:
            self.flatten_observation = False

        # Opponent related variables
        self.battles_won_opponent = 0
        self.battles_defeat = 0
        self._min_unit_type_opponent = 0
        self.marine_id_opponent = self.marauder_id_opponent = self.medivac_id_opponent = 0
        self.hydralisk_id_opponent = self.zergling_id_opponent = self.baneling_id_opponent = 0
        self.stalker_id_opponent = self.colossus_id_opponent = self.zealot_id_opponent = 0
        self.max_distance_x = 0
        self.max_distance_y = 0
        self.map_x = 0
        self.map_y = 0

        self.previous_ally_units = None
        self.previous_enemy_units = None

        self.independent_obs = cfg.independent_obs

        self.action_helper = SMACAction(self.n_agents, self.n_enemies, self.two_player, self.mirror_opponent)
        self.reward_helper = SMACReward(
            self.n_agents,
            self.n_enemies,
            self.two_player,
            self.reward_type,
            self.max_reward,
            reward_only_positive=self.reward_only_positive
        )

        self._observation_space = self.get_obs_space()
        self._action_space = self.action_helper.info(),
        self._reward_space = self.reward_helper.info(),


    def seed(self, seed, dynamic_seed=False):
        self._seed = seed

    def _create_join(self):
        if self.two_player:
            for m in self._maps:
                m.directory = "SMAC_Maps_two_player"
                map_path = m.path
                assert map_path in SUPPORT_MAPS, "We only support the following maps: {}. Please move " \
                                                 "the maps in evaluate/sources/SMAC_Maps_two_player " \
                                                 "to the maps folder of SC2."
        super(SMACEnv, self)._create_join(require_features=False)

    def _get_players(self, game_type, player1_race, player2_race):
        if game_type == 'game_vs_bot':
            agent_num = 1
            print('difficulty', self.difficulty)
            players = [sc2_env.Agent(races[player1_race]), sc2_env.Bot(races[player2_race], self.difficulty)]
        elif game_type == 'agent_vs_agent':
            agent_num = 2
            players = [sc2_env.Agent(races[player1_race]), sc2_env.Agent(races[player2_race])]
        else:
            raise KeyError("invalid game_type: {}".format(game_type))
        return players, agent_num

    def _launch(self):

        print("*****LAUNCH FUNCTION CALLED*****")

        agent_interface_format = sc2_env.parse_agent_interface_format()  # Use all default setting

        SC2Env.__init__(
            self,
            map_name=self.map_name,
            battle_net_map=False,
            players=self.players,
            agent_interface_format=agent_interface_format,
            discount=None,
            discount_zero_after_timeout=False,
            visualize=False,
            step_mul=8,
            realtime=False,
            save_replay_episodes=self.save_replay_episodes,
            replay_dir=None if self.save_replay_episodes is None else ".",
            replay_prefix=None,
            game_steps_per_episode=self.game_steps_per_episode,
            score_index=None,
            score_multiplier=None,
            random_seed=self._seed,
            disable_fog=False,
            ensure_available_actions=True,
            version=None
        )

        self._launch_env_flag = True

        game_info = self._game_info[0]
        map_info = game_info.start_raw
        map_play_area_min = map_info.playable_area.p0
        map_play_area_max = map_info.playable_area.p1
        self.max_distance_x = map_play_area_max.x - map_play_area_min.x
        self.max_distance_y = map_play_area_max.y - map_play_area_min.y
        self.map_x = map_info.map_size.x
        self.map_y = map_info.map_size.y

        self.action_helper.update(map_info, self.map_x, self.map_y)

    def _restart_episode(self):
        """Restart the environment by killing all units on the map.
        There is a trigger in the SC2Map file, which restarts the
        episode when there are no units left.
        """
        try:
            run_commands = [
                (
                    self._controllers[0].debug,
                    d_pb.DebugCommand(
                        kill_unit=d_pb.DebugKillUnit(
                            tag=[unit.tag for unit in self.agents.values() if unit.health > 0] +
                            [unit.tag for unit in self.enemies.values() if unit.health > 0]
                        )
                    )
                )
            ]
            if self.two_player:
                run_commands.append(
                    (self._controllers[1].debug, d_pb.DebugCommand(kill_unit=d_pb.DebugKillUnit(tag=[])))
                )
            # Kill all units on the map.
            self._parallel.run(run_commands)
            # Forward 2 step to make sure all units revive.
            ret = self._parallel.run((c.step, 2) for c in self._controllers)
        except (protocol.ProtocolError, protocol.ConnectionError) as e:
            print("Error happen in _restart. Error: ", e)
            self.full_restart()

    def full_restart(self):
        self.close()
        self._launch()
        self.force_restarts += 1
        self.just_force_restarts = True

    def reset(self):
        self._episode_steps = 0
        self._final_eval_fake_reward = 0.
        old_unit_tags = set(u.tag for u in self.agents.values()).union(set(u.tag for u in self.enemies.values()))

        if self.just_force_restarts:
            old_unit_tags = set()
            self.just_force_restarts = False

        if self._launch_env_flag:
            # Launch StarCraft II
            print("*************LAUNCH TOTAL GAME********************")
            self._launch()
            self._launch_env_flag = False
        elif (self._total_steps > self._next_reset_steps) or (self.save_replay_episodes is not None):
            # Avoid hitting the real episode limit of SC2 env
            print("We are full restarting the environment! save_replay_episodes: ", self.save_replay_episodes)
            self.full_restart()
            old_unit_tags = set()
            self._next_reset_steps += FORCE_RESTART_INTERVAL
        else:
            self._restart_episode()

        # Information kept for counting the reward
        self.win_counted = False
        self.defeat_counted = False

        self.action_helper.reset()

        self.previous_ally_units = None
        self.previous_enemy_units = None

        # if self.heuristic_ai:
        #     self.heuristic_targets = [None] * self.n_agents

        count = 0
        while count <= 5:
            self._update_obs()
            #print("INTERNAL INIT UNIT BEGIN")
            init_flag = self.init_units(old_unit_tags)
            #print("INTERNAL INIT UNIT OVER", init_flag)
            count += 1
            if init_flag:
                break
            else:
                old_unit_tags = set()
        if count >= 5:
            raise RuntimeError("reset 5 times error")

        self.reward_helper.reset(self.max_reward)

        assert all(u.health > 0 for u in self.agents.values())
        assert all(u.health > 0 for u in self.enemies.values())

        if not self.two_player:
            if self.obs_alone:
                agent_state, agent_alone_state, agent_alone_padding_state = self.get_obs()
                return {
                    'agent_state': agent_state,
                    'agent_alone_state': agent_alone_state,
                    'agent_alone_padding_state': agent_alone_padding_state,
                    'global_state': self.get_state(),
                    'action_mask': self.get_avail_actions()
                }
            elif self.independent_obs:
                return {
                    'agent_state': self.get_obs(),
                    'global_state': self.get_obs(),
                    'action_mask': self.get_avail_actions(),
                }
            elif self.special_global_state:
                return {
                    'agent_state': self.get_obs(),
                    'global_state': self.get_global_special_state(),
                    'action_mask': self.get_avail_actions(),
                }
            else:
                return {
                    'agent_state': self.get_obs(),
                    'global_state': self.get_state(),
                    'action_mask': self.get_avail_actions(),
                }

        return {
            'agent_state': {
                ORIGINAL_AGENT: self.get_obs(),
                OPPONENT_AGENT: self.get_obs(True)
            },
            'global_state': {
                ORIGINAL_AGENT: self.get_state(),
                OPPONENT_AGENT: self.get_state(True)
            },
            'action_mask': {
                ORIGINAL_AGENT: self.get_avail_actions(),
                OPPONENT_AGENT: self.get_avail_actions(True),
            },
        }

    def _submit_actions(self, actions):
        if self.two_player:
            # actions is a dict with 'me' and 'opponent' keys.
            actions_me, actions_opponent = actions[ORIGINAL_AGENT], actions[OPPONENT_AGENT]
            self._parallel.run(
                [
                    (self._controllers[0].actions, sc_pb.RequestAction(actions=actions_me)),
                    (self._controllers[1].actions, sc_pb.RequestAction(actions=actions_opponent))
                ]
            )
            step_mul = self._step_mul
            if step_mul <= 0:
                raise ValueError("step_mul should be positive, got {}".format(step_mul))
            if not any(c.status_ended for c in self._controllers):  # May already have ended.
                self._parallel.run((c.step, step_mul) for c in self._controllers)
            self._update_obs(target_game_loop=self._episode_steps + step_mul)
        else:
            # actions is a sequence
            # Send action request
            req_actions = sc_pb.RequestAction(actions=actions)
            self._controllers[0].actions(req_actions)
            self._controllers[0].step(self._step_mul)
            self._update_obs()

    def _get_empty_action(self, old_action):
        me_act = []
        for a_id in range(self.n_agents):
            no_op = self.action_helper.get_avail_agent_actions(a_id, self, is_opponent=False)[0]
            me_act.append(0 if no_op else 1)

        if isinstance(old_action, dict):
            op_act = []
            for a_id in range(self.n_enemies):
                no_op = self.action_helper.get_avail_agent_actions(a_id, self, is_opponent=False)[0]
                op_act.append(0 if no_op else 1)
            new_action = {ORIGINAL_AGENT: me_act, OPPONENT_AGENT: op_act}
        else:
            new_action = me_act
        return new_action

    def step(self, actions, force_return_two_player=False):
        processed_actions = self.action_helper.get_action(actions, self)
        # self._submit_actions(processed_actions)
        try:
            # print("Submitting actions: ", actions)
            self._submit_actions(processed_actions)
            # raise ValueError()  # To test the functionality of restart
        except (protocol.ProtocolError, protocol.ConnectionError, ValueError) as e:
            print("Error happen in step! Error: ", e)
            self.full_restart()
            info = {'abnormal': True}
            return self.SMACTimestep(obs=None, reward=None, done=True, info=info, episode_steps=self._episode_steps)

        # Update units
        game_end_code = self.update_units()
        rewards, terminates, infos = self._collect_step_data(game_end_code, actions)

        infos["draw"] = int(not (infos["me"]["battle_won"] or infos["opponent"]["battle_won"]))

        if (not self.two_player) and (not force_return_two_player):
            rewards, terminates, new_infos = rewards[ORIGINAL_AGENT], terminates[ORIGINAL_AGENT], infos[ORIGINAL_AGENT]
            self._final_eval_fake_reward += rewards
            new_infos["battle_lost"] = infos[OPPONENT_AGENT]["battle_won"]
            new_infos["draw"] = infos["draw"]
            new_infos['final_eval_reward'] = infos['final_eval_reward']
            if 'episode_info' in infos:
                new_infos['episode_info'] = infos['episode_info']
            new_infos['fake_final_eval_reward'] = infos['fake_final_eval_reward']
            infos = new_infos
            if self.obs_alone:
                agent_state, agent_alone_state, agent_alone_padding_state = self.get_obs()
                obs = {
                    'agent_state': agent_state,
                    'agent_alone_state': agent_alone_state,
                    'agent_alone_padding_state': agent_alone_padding_state,
                    'global_state': self.get_state(),
                    'action_mask': self.get_avail_actions()
                }
            elif self.independent_obs:
                obs = {
                    'agent_state': self.get_obs(),
                    'global_state': self.get_obs(),
                    'action_mask': self.get_avail_actions(),
                }
            elif self.special_global_state:
                obs = {
                    'agent_state': self.get_obs(),
                    'global_state': self.get_global_special_state(),
                    'action_mask': self.get_avail_actions(),
                }
            else:
                obs = {
                    'agent_state': self.get_obs(),
                    'global_state': self.get_state(),
                    'action_mask': self.get_avail_actions(),
                }
        else:
            raise NotImplementedError

        return self.SMACTimestep(
            obs=copy.deepcopy(obs), reward=rewards, done=terminates, info=infos, episode_steps=self._episode_steps
        )

    def _collect_step_data(self, game_end_code, action):
        """This function is called only once at each step, no matter whether you take opponent as agent.
        We already return dicts for each term, as in Multi-agent scenario.
        """
        self._total_steps += 1
        self._episode_steps += 1

        terminated = False

        reward = self.reward_helper.get_reward(self, action, game_end_code, self.win_counted, self.defeat_counted)
        for k in reward:
            reward[k] = np.array(reward[k]).astype(np.float32)

        info = {
            ORIGINAL_AGENT: {
                "battle_won": False
            },
            OPPONENT_AGENT: {
                "battle_won": False
            },
            'final_eval_reward': 0.,
            'fake_final_eval_reward': 0.
        }

        if game_end_code is not None:
            # Battle is over
            terminated = True
            self.battles_game += 1
            if game_end_code == 1 and not self.win_counted:
                # The original agent win the game.
                self.battles_won += 1
                self.win_counted = True
                info[ORIGINAL_AGENT]["battle_won"] = True
                info[OPPONENT_AGENT]["battle_won"] = False
                info['final_eval_reward'] = 1.
            elif game_end_code == -1 and not self.defeat_counted:
                self.defeat_counted = True
                info[ORIGINAL_AGENT]["battle_won"] = False
                info[OPPONENT_AGENT]["battle_won"] = True

        elif self._episode_steps >= self.episode_limit:
            # Episode limit reached
            terminated = True
            if self.continuing_episode:
                info[ORIGINAL_AGENT]["episode_limit"] = True
                info[OPPONENT_AGENT]["episode_limit"] = True
            self.battles_game += 1
            self.timeouts += 1
            # info['final_eval_reward'] = -0.5

            # if sum(u.health + u.shield for u in self.agents.values()) >= \
            #         sum(u.health + u.shield for u in self.enemies.values()):
            #     # lj fix
            #     reward[ORIGINAL_AGENT] += 1
            #     reward[OPPONENT_AGENT] += -1
            # else:
            #     reward[ORIGINAL_AGENT] += -1
            #     reward[OPPONENT_AGENT] += 1

        if terminated:
            self._episode_count += 1
            # 1-dim to 0-dim
            # count units that are still alive
            dead_allies, dead_enemies = 0, 0
            for al_id, al_unit in self.agents.items():
                if al_unit.health == 0:
                    dead_allies += 1
            for e_id, e_unit in self.enemies.items():
                if e_unit.health == 0:
                    dead_enemies += 1

            info['episode_info'] = {
                'final_eval_fake_reward': self._final_eval_fake_reward[0],
                'dead_allies': dead_allies,
                'dead_enemies': dead_enemies
            }
            self._final_eval_fake_reward = 0.

        # PZH: Zero at first step
        if self._episode_steps == 1:
            for k in reward.keys():
                reward[k] *= 0.0
            if terminated:
                print("WARNNING! Should not terminate at the first step!")

        # Test purpose
        # reward = {k: 0 * v + 100 for k, v in reward.items()}
        info['fake_final_eval_reward'] = reward[ORIGINAL_AGENT]
        return reward, {ORIGINAL_AGENT: terminated, OPPONENT_AGENT: terminated, "__all__": terminated}, info

    def close(self):
        SC2Env.close(self)

    def init_units(self, old_unit_tags):
        count = 0
        while count < 10:
            # Sometimes not all units have yet been created by SC2
            self.agents = {}
            self.enemies = {}

            ally_units = [
                unit for unit in self._obs.observation.raw_data.units
                if (unit.owner == 1) and (unit.tag not in old_unit_tags)
            ]
            ally_units_sorted = sorted(
                ally_units,
                key=attrgetter("unit_type", "pos.x", "pos.y"),
                reverse=False,
            )

            for i in range(len(ally_units_sorted)):
                self.agents[i] = ally_units_sorted[i]

            self.max_reward = self.n_enemies * self.reward_death_value + self.reward_win
            for unit in self._obs.observation.raw_data.units:
                if (unit.owner == 2) and (unit.tag not in old_unit_tags):
                    self.enemies[len(self.enemies)] = unit
                    # if self._episode_count == 0:
                    self.max_reward += unit.health_max + unit.shield_max

            all_agents_created = (len(self.agents) == self.n_agents)
            all_enemies_created = (len(self.enemies) == self.n_enemies)

            all_agents_health = all(u.health > 0 for u in self.agents.values())
            all_enemies_health = all(u.health > 0 for u in self.enemies.values())

            if all_agents_created and all_enemies_created \
                    and all_agents_health and all_enemies_health:  # all good
                if self._episode_count == 0:
                    min_unit_type = min(unit.unit_type for unit in self.agents.values())
                    min_unit_type_opponent = min(unit.unit_type for unit in self.enemies.values())
                    self._init_ally_unit_types(min_unit_type)
                    self._init_enemy_unit_types(min_unit_type_opponent)
                return True
            else:
                print(
                    "***ALL GOOD FAIL***", all_agents_created, all_enemies_created, all_agents_health,
                    all_enemies_health, len(self._obs.observation.raw_data.units)
                )
                print(
                    (len(self.agents) == self.n_agents), (len(self.enemies) == self.n_enemies), len(self.agents),
                    self.n_agents, len(self.enemies), self.n_enemies
                )
                self._restart_episode()
                count += 1

            try:
                self._parallel.run((c.step, 1) for c in self._controllers)
                self._update_obs()

            except (protocol.ProtocolError, protocol.ConnectionError) as e:
                print("Error happen in init_units.", e)
                self.full_restart()
                return False
        if count >= 10:
            self.full_restart()
            return False

    def _init_enemy_unit_types(self, min_unit_type_opponent):
        """Initialise ally unit types. Should be called once from the
        init_units function.
        """
        self._min_unit_type_opponent = min_unit_type_opponent
        if self.map_type == "marines":
            self.marine_id_opponent = min_unit_type_opponent
        elif self.map_type == "stalkers_and_zealots":
            self.stalker_id_opponent = min_unit_type_opponent
            self.zealot_id_opponent = min_unit_type_opponent + 1
        elif self.map_type == "colossi_stalkers_zealots":
            self.colossus_id_opponent = min_unit_type_opponent
            self.stalker_id_opponent = min_unit_type_opponent + 1
            self.zealot_id_opponent = min_unit_type_opponent + 2
        elif self.map_type == "MMM":
            self.marauder_id_opponent = min_unit_type_opponent
            self.marine_id_opponent = min_unit_type_opponent + 1
            self.medivac_id_opponent = min_unit_type_opponent + 2
        elif self.map_type == "zealots":
            self.zealot_id_opponent = min_unit_type_opponent
        elif self.map_type == "hydralisks":
            self.hydralisk_id_opponent = min_unit_type_opponent
        elif self.map_type == "stalkers":
            self.stalker_id_opponent = min_unit_type_opponent
        elif self.map_type == "colossus":
            self.colossus_id_opponent = min_unit_type_opponent
        elif self.map_type == "bane":
            self.baneling_id_opponent = min_unit_type_opponent
            self.zergling_id_opponent = min_unit_type_opponent + 1

    # ================
    def unit_max_shield(self, unit, is_opponent=False):
        """Returns maximal shield for a given unit."""
        stalker_id = self.stalker_id_opponent if is_opponent else self.stalker_id
        zealot_id = self.zealot_id_opponent if is_opponent else self.zealot_id
        colossus_id = self.colossus_id_opponent if is_opponent else self.colossus_id
        if unit.unit_type == 74 or unit.unit_type == stalker_id:
            return 80  # Protoss's Stalker
        if unit.unit_type == 73 or unit.unit_type == zealot_id:
            return 50  # Protoss's Zaelot
        if unit.unit_type == 4 or unit.unit_type == colossus_id:
            return 150  # Protoss's Colossus

    def get_unit_type_id(self, unit, ally, is_opponent=False):
        if is_opponent and ally:
            return unit.unit_type - self._min_unit_type_opponent
        else:
            if ally:  # use new SC2 unit types
                if self.map_type == "infestor_viper":
                    if unit.unit_type == 393:
                        type_id = 0
                    else:
                        type_id = 1
                else:
                    type_id = unit.unit_type - self._min_unit_type
            else:  # use default SC2 unit types
                if self.map_type == "stalkers_and_zealots":
                    # id(Stalker) = 74, id(Zealot) = 73
                    type_id = unit.unit_type - 73
                elif self.map_type == "colossi_stalkers_zealots":
                    # id(Stalker) = 74, id(Zealot) = 73, id(Colossus) = 4
                    if unit.unit_type == 4:
                        type_id = 0
                    elif unit.unit_type == 74:
                        type_id = 1
                    else:
                        type_id = 2
                elif self.map_type == "bane":
                    if unit.unit_type == 9:
                        type_id = 0
                    else:
                        type_id = 1
                elif self.map_type == "MMM":
                    if unit.unit_type == 51:
                        type_id = 0
                    elif unit.unit_type == 48:
                        type_id = 1
                    else:
                        type_id = 2
                elif self.map_type == "infestor_viper":
                    if unit.unit_type == 393:
                        type_id = 0
                    else:
                        type_id = 1
                else:
                    raise ValueError()
            return type_id

    def _update_obs(self, target_game_loop=0):
        # Transform in the thread so it runs while waiting for other observations.
        # def parallel_observe(c, f):

        if self.two_player:

            def parallel_observe(c):
                obs = c.observe(target_game_loop=target_game_loop)
                # agent_obs = f.transform_obs(obs)
                return obs

            # with self._metrics.measure_observation_time():
            self._obses = self._parallel.run((parallel_observe, c) for c in self._controllers)
        else:
            self._obses = [self._controllers[0].observe()]

        self._obs = self._obses[0]

    def _init_ally_unit_types(self, min_unit_type):
        """Initialise ally unit types. Should be called once from the
        init_units function.
        """
        self._min_unit_type = min_unit_type
        if self.map_type == "marines":
            self.marine_id = min_unit_type
        elif self.map_type == "stalkers_and_zealots":
            self.stalker_id = min_unit_type
            self.zealot_id = min_unit_type + 1
        elif self.map_type == "colossi_stalkers_zealots":
            self.colossus_id = min_unit_type
            self.stalker_id = min_unit_type + 1
            self.zealot_id = min_unit_type + 2
        elif self.map_type == "MMM":
            self.marauder_id = min_unit_type
            self.marine_id = min_unit_type + 1
            self.medivac_id = min_unit_type + 2
        elif self.map_type == "zealots":
            self.zealot_id = min_unit_type
        elif self.map_type == "hydralisks":
            self.hydralisk_id = min_unit_type
        elif self.map_type == "stalkers":
            self.stalker_id = min_unit_type
        elif self.map_type == "colossus":
            self.colossus_id = min_unit_type
        elif self.map_type == "bane":
            self.baneling_id = min_unit_type
            self.zergling_id = min_unit_type + 1

    def get_obs(self, is_opponent=False):
        """Returns all agent observations in a list.
        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        agents_obs_list = [self.get_obs_agent(i, is_opponent) for i in range(self.n_agents)]

        if self.mirror_opponent and is_opponent:
            assert not self.flatten_observation
            new_obs = list()
            for agent_obs in agents_obs_list:
                new_agent_obs = dict()
                for key, feat in agent_obs.items():
                    feat = feat.copy()

                    if key == "move_feats":
                        can_move_right = feat[2]
                        can_move_left = feat[3]
                        feat[3] = can_move_right
                        feat[2] = can_move_left

                    elif key == "enemy_feats" or key == "ally_feats":
                        for unit_id in range(feat.shape[0]):
                            # Relative x
                            feat[unit_id, 2] = -feat[unit_id, 2]

                    new_agent_obs[key] = feat
                new_obs.append(new_agent_obs)
            agents_obs_list = new_obs

        if not self.flatten_observation:
            agents_obs_list = self._flatten_obs(agents_obs_list)
        if self.obs_alone:
            agents_obs_list, agents_obs_alone_list, agents_obs_alone_padding_list = list(zip(*agents_obs_list))
            return np.array(agents_obs_list).astype(np.float32), np.array(agents_obs_alone_list).astype(
                np.float32
            ), np.array(agents_obs_alone_padding_list).astype(np.float32)
        else:
            return np.array(agents_obs_list).astype(np.float32)

    def get_obs_agent(self, agent_id, is_opponent=False):
        unit = self.get_unit_by_id(agent_id, is_opponent=is_opponent)

        # TODO All these function should have an opponent version
        enemy_feats_dim = self.get_obs_enemy_feats_size()
        ally_feats_dim = self.get_obs_ally_feats_size()
        own_feats_dim = self.get_obs_own_feats_size()

        enemy_feats = np.zeros(enemy_feats_dim, dtype=np.float32)
        ally_feats = np.zeros(ally_feats_dim, dtype=np.float32)
        own_feats = np.zeros(own_feats_dim, dtype=np.float32)

        move_feats = self.action_helper.get_movement_features(agent_id, self, is_opponent)

        if unit.health > 0:  # otherwise dead, return all zeros
            x = unit.pos.x
            y = unit.pos.y
            sight_range = self.unit_sight_range(agent_id)
            avail_actions = self.action_helper.get_avail_agent_actions(agent_id, self, is_opponent)

            # Enemy features
            if is_opponent:
                enemy_items = self.agents.items()
            else:
                enemy_items = self.enemies.items()
            for e_id, e_unit in enemy_items:
                e_x = e_unit.pos.x
                e_y = e_unit.pos.y
                dist = distance(x, y, e_x, e_y)

                if (dist < sight_range and e_unit.health > 0):  # visible and alive
                    # Sight range > shoot range
                    enemy_feats[e_id, 0] = avail_actions[self.action_helper.n_actions_no_attack + e_id]  # available
                    enemy_feats[e_id, 1] = dist / sight_range  # distance
                    enemy_feats[e_id, 2] = (e_x - x) / sight_range  # relative X
                    enemy_feats[e_id, 3] = (e_y - y) / sight_range  # relative Y

                    ind = 4
                    if self.obs_all_health:
                        enemy_feats[e_id, ind] = (e_unit.health / e_unit.health_max)  # health
                        ind += 1
                        if self.shield_bits_enemy > 0:
                            max_shield = self.unit_max_shield(e_unit, not is_opponent)
                            enemy_feats[e_id, ind] = (e_unit.shield / max_shield)  # shield
                            ind += 1

                    if self.unit_type_bits > 0:
                        # If enemy is computer, than use ally=False, but since now we use
                        #  agent for enemy, ally=True
                        if self.two_player:
                            type_id = self.get_unit_type_id(e_unit, True, not is_opponent)
                        else:
                            type_id = self.get_unit_type_id(e_unit, False, False)
                        enemy_feats[e_id, ind + type_id] = 1  # unit type

            # Ally features
            al_ids = [
                al_id for al_id in range((self.n_agents if not is_opponent else self.n_enemies)) if al_id != agent_id
            ]
            for i, al_id in enumerate(al_ids):

                al_unit = self.get_unit_by_id(al_id, is_opponent=is_opponent)
                al_x = al_unit.pos.x
                al_y = al_unit.pos.y
                dist = distance(x, y, al_x, al_y)

                if (dist < sight_range and al_unit.health > 0):  # visible and alive
                    ally_feats[i, 0] = 1  # visible
                    ally_feats[i, 1] = dist / sight_range  # distance
                    ally_feats[i, 2] = (al_x - x) / sight_range  # relative X
                    ally_feats[i, 3] = (al_y - y) / sight_range  # relative Y

                    ind = 4
                    if self.obs_all_health:
                        ally_feats[i, ind] = (al_unit.health / al_unit.health_max)  # health
                        ind += 1
                        if self.shield_bits_ally > 0:
                            max_shield = self.unit_max_shield(al_unit, is_opponent)
                            ally_feats[i, ind] = (al_unit.shield / max_shield)  # shield
                            ind += 1

                    if self.unit_type_bits > 0:
                        type_id = self.get_unit_type_id(al_unit, True, is_opponent)
                        ally_feats[i, ind + type_id] = 1
                        ind += self.unit_type_bits

                    # LJ fix
                    # if self.obs_last_action:
                    #     ally_feats[i, ind:] = self.action_helper.get_last_action(is_opponent)[al_id]

            # Own features
            ind = 0
            if self.obs_own_health:
                own_feats[ind] = unit.health / unit.health_max
                ind += 1
                if self.shield_bits_ally > 0:
                    max_shield = self.unit_max_shield(unit, is_opponent)
                    own_feats[ind] = unit.shield / max_shield
                    ind += 1

            if self.unit_type_bits > 0:
                type_id = self.get_unit_type_id(unit, True, is_opponent)
                own_feats[ind + type_id] = 1
                ind += self.unit_type_bits
            if self.obs_last_action:
                own_feats[ind:] = self.action_helper.get_last_action(is_opponent)[agent_id]

        if is_opponent:
            agent_id_feats = np.zeros(self.n_enemies)
        else:
            agent_id_feats = np.zeros(self.n_agents)
        agent_id_feats[agent_id] = 1
        # Only set to false by outside wrapper
        if self.flatten_observation:
            agent_obs = np.concatenate(
                (
                    move_feats.flatten(),
                    enemy_feats.flatten(),
                    ally_feats.flatten(),
                    own_feats.flatten(),
                    agent_id_feats,
                )
            )
            if self.obs_timestep_number:
                agent_obs = np.append(agent_obs, self._episode_steps / self.episode_limit)
            if self.obs_alone:
                agent_obs_alone = np.concatenate(
                    (
                        move_feats.flatten(),
                        enemy_feats.flatten(),
                        own_feats.flatten(),
                        agent_id_feats,
                    )
                )
                agent_obs_alone_padding = np.concatenate(
                    (
                        move_feats.flatten(),
                        enemy_feats.flatten(),
                        np.zeros_like(ally_feats.flatten()),
                        own_feats.flatten(),
                        agent_id_feats,
                    )
                )
                if self.obs_timestep_number:
                    agent_obs_alone = np.append(agent_obs_alone, self._episode_steps / self.episode_limit)
                    agent_obs_alone_padding = np.append(
                        agent_obs_alone_padding, self._episode_steps / self.episode_limit
                    )
                return agent_obs, agent_obs_alone, agent_obs_alone_padding
            else:
                return agent_obs
        else:
            agent_obs = dict(
                move_feats=move_feats,
                enemy_feats=enemy_feats,
                ally_feats=ally_feats,
                own_feats=own_feats,
                agent_id_feats=agent_id_feats
            )
            if self.obs_timestep_number:
                agent_obs["obs_timestep_number"] = self._episode_steps / self.episode_limit

        return agent_obs

    def get_unit_by_id(self, a_id, is_opponent=False):
        """Get unit by ID."""
        if is_opponent:
            return self.enemies[a_id]
        return self.agents[a_id]

    def get_obs_enemy_feats_size(self):
        """ Returns the dimensions of the matrix containing enemy features.
        Size is n_enemies x n_features.
        """
        nf_en = 4 + self.unit_type_bits

        if self.obs_all_health:
            nf_en += 1 + self.shield_bits_enemy

        return self.n_enemies, nf_en

    def get_obs_ally_feats_size(self):
        """Returns the dimensions of the matrix containing ally features.
        Size is n_allies x n_features.
        """
        nf_al = 4 + self.unit_type_bits

        if self.obs_all_health:
            nf_al += 1 + self.shield_bits_ally

        # LJ fix
        # if self.obs_last_action:
        #     nf_al += self.n_actions

        return self.n_agents - 1, nf_al

    def get_obs_own_feats_size(self):
        """Returns the size of the vector containing the agents' own features.
        """
        own_feats = self.unit_type_bits
        if self.obs_own_health:
            own_feats += 1 + self.shield_bits_ally
        if self.obs_timestep_number:
            own_feats += 1
        if self.obs_last_action:
            own_feats += self.n_actions

        return own_feats

    def get_obs_move_feats_size(self):
        """Returns the size of the vector containing the agents's movement-related features."""
        return self.action_helper.get_obs_move_feats_size()

    def get_state_size(self, is_opponent=False):
        """Returns the size of the global state."""
        if self.obs_instead_of_state:
            return self.get_obs_size(is_opponent) * self.n_agents

        nf_al = 4 + self.shield_bits_ally + self.unit_type_bits
        nf_en = 3 + self.shield_bits_enemy + self.unit_type_bits

        enemy_state = self.n_enemies * nf_en
        ally_state = self.n_agents * nf_al

        size = enemy_state + ally_state

        if self.state_last_action:
            if is_opponent:
                size += self.n_enemies * self.n_actions_opponent
            else:
                size += self.n_agents * self.n_actions
        if self.state_timestep_number:
            size += 1

        return size

    def get_obs_size(self, is_opponent=False):
        # TODO suppose the agents formation are same for both opponent and me. This can be extended in future.
        """Returns the size of the observation."""
        own_feats = self.get_obs_own_feats_size()
        move_feats = self.get_obs_move_feats_size()

        n_enemies, n_enemy_feats = self.get_obs_enemy_feats_size()
        n_allies, n_ally_feats = self.get_obs_ally_feats_size()

        enemy_feats = n_enemies * n_enemy_feats
        ally_feats = n_allies * n_ally_feats

        if is_opponent:
            agent_id_feats = self.n_enemies
        else:
            agent_id_feats = self.n_agents
        return move_feats + enemy_feats + ally_feats + own_feats + agent_id_feats

    def get_obs_alone_size(self, is_opponent=False):
        # TODO suppose the agents formation are same for both opponent and me. This can be extended in future.
        """Returns the size of the observation."""
        own_feats = self.get_obs_own_feats_size()
        move_feats = self.get_obs_move_feats_size()

        n_enemies, n_enemy_feats = self.get_obs_enemy_feats_size()

        enemy_feats = n_enemies * n_enemy_feats

        if is_opponent:
            agent_id_feats = self.n_enemies
        else:
            agent_id_feats = self.n_agents
        return move_feats + enemy_feats + own_feats + agent_id_feats

    def get_state(self, is_opponent=False):
        if self.obs_instead_of_state:
            obs_concat = np.concatenate(self.get_obs(), axis=0).astype(np.float32)
            return obs_concat

        nf_al = 4 + self.shield_bits_ally + self.unit_type_bits
        nf_en = 3 + self.shield_bits_enemy + self.unit_type_bits

        ally_state = np.zeros((self.n_agents, nf_al))
        enemy_state = np.zeros((self.n_enemies, nf_en))

        center_x = self.map_x / 2
        center_y = self.map_y / 2

        if is_opponent:
            iterator = self.enemies.items()
        else:
            iterator = self.agents.items()

        for al_id, al_unit in iterator:
            if al_unit.health > 0:
                x = al_unit.pos.x
                y = al_unit.pos.y
                max_cd = self.unit_max_cooldown(al_unit, is_opponent=is_opponent)

                ally_state[al_id, 0] = (al_unit.health / al_unit.health_max)  # health
                if (self.map_type == "MMM"
                        and al_unit.unit_type == (self.medivac_id_opponent if is_opponent else self.medivac_id)):
                    ally_state[al_id, 1] = al_unit.energy / max_cd  # energy
                else:
                    ally_state[al_id, 1] = (al_unit.weapon_cooldown / max_cd)  # cooldown
                ally_state[al_id, 2] = (x - center_x) / self.max_distance_x  # relative X
                ally_state[al_id, 3] = (y - center_y) / self.max_distance_y  # relative Y

                ind = 4
                if self.shield_bits_ally > 0:
                    max_shield = self.unit_max_shield(al_unit, is_opponent=is_opponent)
                    ally_state[al_id, ind] = (al_unit.shield / max_shield)  # shield
                    ind += 1

                if self.unit_type_bits > 0:
                    type_id = self.get_unit_type_id(al_unit, True, is_opponent=is_opponent)
                    ally_state[al_id, ind + type_id] = 1

        if is_opponent:
            iterator = self.agents.items()
        else:
            iterator = self.enemies.items()
        for e_id, e_unit in iterator:
            if e_unit.health > 0:
                x = e_unit.pos.x
                y = e_unit.pos.y

                enemy_state[e_id, 0] = (e_unit.health / e_unit.health_max)  # health
                enemy_state[e_id, 1] = (x - center_x) / self.max_distance_x  # relative X
                enemy_state[e_id, 2] = (y - center_y) / self.max_distance_y  # relative Y

                ind = 3
                if self.shield_bits_enemy > 0:
                    max_shield = self.unit_max_shield(e_unit, is_opponent=False)
                    enemy_state[e_id, ind] = (e_unit.shield / max_shield)  # shield
                    ind += 1

                if self.unit_type_bits > 0:
                    type_id = self.get_unit_type_id(e_unit, True if self.two_player else False, is_opponent=False)
                    enemy_state[e_id, ind + type_id] = 1

        last_action = self.action_helper.get_last_action(is_opponent)
        if self.flatten_observation:
            state = np.append(ally_state.flatten(), enemy_state.flatten())
            if self.state_last_action:
                state = np.append(state, last_action.flatten())
            if self.state_timestep_number:
                state = np.append(state, self._episode_steps / self.episode_limit)
            state = state.astype(dtype=np.float32)
        else:
            state = dict(ally_state=ally_state, enemy_state=enemy_state)
            if self.state_last_action:
                state["last_action"] = last_action
            if self.state_timestep_number:
                state["state_timestep_number"] = self._episode_steps / self.episode_limit

        if self.mirror_opponent and is_opponent:
            assert not self.flatten_observation

            new_state = dict()
            for key, s in state.items():
                s = s.copy()

                if key == "ally_state":
                    # relative x
                    for unit_id in range(s.shape[0]):
                        s[unit_id, 2] = -s[unit_id, 2]

                elif key == "enemy_state":
                    # relative x
                    for unit_id in range(s.shape[0]):
                        s[unit_id, 1] = -s[unit_id, 1]

                # key == "last_action" is processed in SMACAction
                new_state[key] = s
            state = new_state

        if not self.flatten_observation:
            state = self._flatten_state(state)
        return np.array(state).astype(np.float32)

    def get_global_special_state(self, is_opponent=False):
        """Returns all agent observations in a list.
        NOTE: Agents should have access only to their local observations
        during decentralised execution.
        """
        agents_obs_list = [self.get_state_agent(i, is_opponent) for i in range(self.n_agents)]

        return np.array(agents_obs_list).astype(np.float32)

    def get_global_special_state_size(self, is_opponent=False):
        enemy_feats_dim = self.get_state_enemy_feats_size()
        ally_feats_dim = self.get_state_ally_feats_size()
        own_feats_dim = self.get_state_own_feats_size()
        size = enemy_feats_dim + ally_feats_dim + own_feats_dim + self.n_agents
        if self.state_timestep_number:
            size += 1
        return size

    def get_state_agent(self, agent_id, is_opponent=False):
        """Returns observation for agent_id. The observation is composed of:

           - agent movement features (where it can move to, height information and pathing grid)
           - enemy features (available_to_attack, health, relative_x, relative_y, shield, unit_type)
           - ally features (visible, distance, relative_x, relative_y, shield, unit_type)
           - agent unit features (health, shield, unit_type)

           All of this information is flattened and concatenated into a list,
           in the aforementioned order. To know the sizes of each of the
           features inside the final list of features, take a look at the
           functions ``get_obs_move_feats_size()``,
           ``get_obs_enemy_feats_size()``, ``get_obs_ally_feats_size()`` and
           ``get_obs_own_feats_size()``.

           The size of the observation vector may vary, depending on the
           environment configuration and type of units present in the map.
           For instance, non-Protoss units will not have shields, movement
           features may or may not include terrain height and pathing grid,
           unit_type is not included if there is only one type of unit in the
           map etc.).

           NOTE: Agents should have access only to their local observations
           during decentralised execution.
        """
        if self.obs_instead_of_state:
            obs_concat = np.concatenate(self.get_obs(), axis=0).astype(np.float32)
            return obs_concat

        unit = self.get_unit_by_id(agent_id)

        enemy_feats_dim = self.get_state_enemy_feats_size()
        ally_feats_dim = self.get_state_ally_feats_size()
        own_feats_dim = self.get_state_own_feats_size()

        enemy_feats = np.zeros(enemy_feats_dim, dtype=np.float32)
        ally_feats = np.zeros(ally_feats_dim, dtype=np.float32)
        own_feats = np.zeros(own_feats_dim, dtype=np.float32)
        agent_id_feats = np.zeros(self.n_agents, dtype=np.float32)

        center_x = self.map_x / 2
        center_y = self.map_y / 2

        if (self.death_mask and unit.health > 0) or (not self.death_mask):  # otherwise dead, return all zeros
            x = unit.pos.x
            y = unit.pos.y
            sight_range = self.unit_sight_range(agent_id)
            last_action = self.action_helper.get_last_action(is_opponent)

            # Movement features
            avail_actions = self.get_avail_agent_actions(agent_id)

            # Enemy features
            for e_id, e_unit in self.enemies.items():
                e_x = e_unit.pos.x
                e_y = e_unit.pos.y
                dist = self.distance(x, y, e_x, e_y)

                if e_unit.health > 0:  # visible and alive
                    # Sight range > shoot range
                    if unit.health > 0:
                        enemy_feats[e_id, 0] = avail_actions[self.action_helper.n_actions_no_attack + e_id]  # available
                        enemy_feats[e_id, 1] = dist / sight_range  # distance
                        enemy_feats[e_id, 2] = (e_x - x) / sight_range  # relative X
                        enemy_feats[e_id, 3] = (e_y - y) / sight_range  # relative Y
                        if dist < sight_range:
                            enemy_feats[e_id, 4] = 1  # visible

                    ind = 5
                    if self.obs_all_health:
                        enemy_feats[e_id, ind] = (e_unit.health / e_unit.health_max)  # health
                        ind += 1
                        if self.shield_bits_enemy > 0:
                            max_shield = self.unit_max_shield(e_unit)
                            enemy_feats[e_id, ind] = (e_unit.shield / max_shield)  # shield
                            ind += 1

                    if self.unit_type_bits > 0:
                        type_id = self.get_unit_type_id(e_unit, False)
                        enemy_feats[e_id, ind + type_id] = 1  # unit type
                        ind += self.unit_type_bits

                    if self.add_center_xy:
                        enemy_feats[e_id, ind] = (e_x - center_x) / self.max_distance_x  # center X
                        enemy_feats[e_id, ind + 1] = (e_y - center_y) / self.max_distance_y  # center Y

            # Ally features
            al_ids = [al_id for al_id in range(self.n_agents) if al_id != agent_id]
            for i, al_id in enumerate(al_ids):

                al_unit = self.get_unit_by_id(al_id)
                al_x = al_unit.pos.x
                al_y = al_unit.pos.y
                dist = self.distance(x, y, al_x, al_y)
                max_cd = self.unit_max_cooldown(al_unit)

                if al_unit.health > 0:  # visible and alive
                    if unit.health > 0:
                        if dist < sight_range:
                            ally_feats[i, 0] = 1  # visible
                        ally_feats[i, 1] = dist / sight_range  # distance
                        ally_feats[i, 2] = (al_x - x) / sight_range  # relative X
                        ally_feats[i, 3] = (al_y - y) / sight_range  # relative Y

                    if (self.map_type == "MMM" and al_unit.unit_type == self.medivac_id):
                        ally_feats[i, 4] = al_unit.energy / max_cd  # energy
                    else:
                        ally_feats[i, 4] = (al_unit.weapon_cooldown / max_cd)  # cooldown

                    ind = 5
                    if self.obs_all_health:
                        ally_feats[i, ind] = (al_unit.health / al_unit.health_max)  # health
                        ind += 1
                        if self.shield_bits_ally > 0:
                            max_shield = self.unit_max_shield(al_unit)
                            ally_feats[i, ind] = (al_unit.shield / max_shield)  # shield
                            ind += 1

                    if self.add_center_xy:
                        ally_feats[i, ind] = (al_x - center_x) / self.max_distance_x  # center X
                        ally_feats[i, ind + 1] = (al_y - center_y) / self.max_distance_y  # center Y
                        ind += 2

                    if self.unit_type_bits > 0:
                        type_id = self.get_unit_type_id(al_unit, True)
                        ally_feats[i, ind + type_id] = 1
                        ind += self.unit_type_bits

                    if self.state_last_action:
                        ally_feats[i, ind:] = last_action[al_id]

            # Own features
            ind = 0
            own_feats[0] = 1  # visible
            own_feats[1] = 0  # distance
            own_feats[2] = 0  # X
            own_feats[3] = 0  # Y
            ind = 4
            if self.obs_own_health:
                own_feats[ind] = unit.health / unit.health_max
                ind += 1
                if self.shield_bits_ally > 0:
                    max_shield = self.unit_max_shield(unit)
                    own_feats[ind] = unit.shield / max_shield
                    ind += 1

            if self.add_center_xy:
                own_feats[ind] = (x - center_x) / self.max_distance_x  # center X
                own_feats[ind + 1] = (y - center_y) / self.max_distance_y  # center Y
                ind += 2

            if self.unit_type_bits > 0:
                type_id = self.get_unit_type_id(unit, True)
                own_feats[ind + type_id] = 1
                ind += self.unit_type_bits

            if self.state_last_action:
                own_feats[ind:] = last_action[agent_id]

        state = np.concatenate((ally_feats.flatten(), enemy_feats.flatten(), own_feats.flatten()))

        # Agent id features
        if self.state_agent_id:
            agent_id_feats[agent_id] = 1.
            state = np.append(state, agent_id_feats.flatten())

        if self.state_timestep_number:
            state = np.append(state, self._episode_steps / self.episode_limit)

        return state

    def get_state_enemy_feats_size(self):
        """ Returns the dimensions of the matrix containing enemy features.
        Size is n_enemies x n_features.
        """
        nf_en = 5 + self.unit_type_bits

        if self.obs_all_health:
            nf_en += 1 + self.shield_bits_enemy

        if self.add_center_xy:
            nf_en += 2

        return self.n_enemies, nf_en

    def get_state_ally_feats_size(self):
        """Returns the dimensions of the matrix containing ally features.
        Size is n_allies x n_features.
        """
        nf_al = 5 + self.unit_type_bits

        if self.obs_all_health:
            nf_al += 1 + self.shield_bits_ally

        if self.state_last_action:
            nf_al += self.n_actions

        if self.add_center_xy:
            nf_al += 2

        return self.n_agents - 1, nf_al

    def get_state_own_feats_size(self):
        """Returns the size of the vector containing the agents' own features.
        """
        own_feats = 4 + self.unit_type_bits
        if self.obs_own_health:
            own_feats += 1 + self.shield_bits_ally

        if self.state_last_action:
            own_feats += self.n_actions

        if self.add_center_xy:
            own_feats += 2

        return own_feats

    @staticmethod
    def distance(x1, y1, x2, y2):
        """Distance between two points."""
        return math.hypot(x2 - x1, y2 - y1)

    def unit_max_cooldown(self, unit, is_opponent=False):
        """Returns the maximal cooldown for a unit."""
        if is_opponent:
            switcher = {
                self.marine_id_opponent: 15,
                self.marauder_id_opponent: 25,
                self.medivac_id_opponent: 200,  # max energy
                self.stalker_id_opponent: 35,
                self.zealot_id_opponent: 22,
                self.colossus_id_opponent: 24,
                self.hydralisk_id_opponent: 10,
                self.zergling_id_opponent: 11,
                self.baneling_id_opponent: 1
            }
        else:
            switcher = {
                self.marine_id: 15,
                self.marauder_id: 25,
                self.medivac_id: 200,  # max energy
                self.stalker_id: 35,
                self.zealot_id: 22,
                self.colossus_id: 24,
                self.hydralisk_id: 10,
                self.zergling_id: 11,
                self.baneling_id: 1
            }
        return switcher.get(unit.unit_type, 15)

    def update_units(self):
        """Update units after an environment step.
        This function assumes that self._obs is up-to-date.
        """
        n_ally_alive = 0
        n_enemy_alive = 0

        # Store previous state
        self.previous_ally_units = copy.deepcopy(self.agents)
        self.previous_enemy_units = copy.deepcopy(self.enemies)

        for al_id, al_unit in self.agents.items():
            updated = False
            for unit in self._obs.observation.raw_data.units:
                if al_unit.tag == unit.tag:
                    self.agents[al_id] = unit
                    updated = True
                    n_ally_alive += 1
                    break

            if not updated:  # dead
                al_unit.health = 0

        for e_id, e_unit in self.enemies.items():
            updated = False
            for unit in self._obs.observation.raw_data.units:
                if e_unit.tag == unit.tag:
                    self.enemies[e_id] = unit
                    updated = True
                    n_enemy_alive += 1
                    break

            if not updated:  # dead
                e_unit.health = 0

        if (n_ally_alive == 0 and n_enemy_alive > 0 or self.only_medivac_left(ally=True)):
            return -1  # lost
        if (n_ally_alive > 0 and n_enemy_alive == 0 or self.only_medivac_left(ally=False)):
            return 1  # won
        if n_ally_alive == 0 and n_enemy_alive == 0:
            return 0

        return None

    def only_medivac_left(self, ally):
        """Check if only Medivac units are left."""
        if self.map_type != "MMM":
            return False

        if ally:
            units_alive = [
                a for a in self.agents.values()
                if (a.health > 0 and a.unit_type != self.medivac_id and a.unit_type != self.medivac_id_opponent
                    )  # <<== add medivac_id_opponent
            ]
            if len(units_alive) == 0:
                return True
            return False
        else:
            units_alive = [
                a for a in self.enemies.values()
                if (a.health > 0 and a.unit_type != self.medivac_id and a.unit_type != self.medivac_id_opponent)
            ]
            if len(units_alive) == 1 and units_alive[0].unit_type == 54:
                return True
            return False

    @property
    def n_actions(self):
        return self.action_helper.n_actions

    @property
    def n_actions_opponent(self):
        return self.n_actions

    # Workaround
    def get_avail_agent_actions(self, agent_id, is_opponent=False):
        return self.action_helper.get_avail_agent_actions(agent_id, self, is_opponent)

    def unit_sight_range(self, agent_id=None):
        """Returns the sight range for an agent."""
        return 9

    @staticmethod
    def _flatten_obs(obs):

        def _get_keys(agent_obs):
            keys = ["move_feats", "enemy_feats", "ally_feats", "own_feats", "agent_id_feats"]
            if "obs_timestep_number" in agent_obs:
                keys.append("obs_timestep_number")
            return keys

        return _flatten(obs, _get_keys)

    @staticmethod
    def _flatten_state(state):

        def _get_keys(s):
            keys = ["ally_state", "enemy_state"]
            if "last_action" in s:
                keys.append("last_action")
            if "state_timestep_number" in s:
                keys.append("state_timestep_number")
            return keys

        return _flatten([state], _get_keys)[0]

    def get_avail_actions(self, is_opponent=False):
        ava_action = self.action_helper.get_avail_actions(self, is_opponent)
        ava_action = np.array(ava_action).astype(np.float32)
        return ava_action

    def get_obs_space(self, is_opponent=False):
        T = EnvElementInfo
        agent_num = self.n_enemies if is_opponent else self.n_agents
        if self.obs_alone:
            obs_space = T(
                {
                    'agent_state': (agent_num, self.get_obs_size(is_opponent)),
                    'agent_alone_state': (agent_num, self.get_obs_alone_size(is_opponent)),
                    'agent_alone_padding_state': (agent_num, self.get_obs_size(is_opponent)),
                    'global_state': (self.get_state_size(is_opponent), ),
                    'action_mask': (agent_num, *self.action_helper.info().shape),
                },
                None,
            )
        else:
            if self.special_global_state:
                obs_space = T(
                    {
                        'agent_state': (agent_num, self.get_obs_size(is_opponent)),
                        'global_state': (agent_num, self.get_global_special_state_size(is_opponent)),
                        'action_mask': (agent_num, *self.action_helper.info().shape),
                    },
                    None,
                )
            else:
                obs_space = T(
                    {
                        'agent_state': (agent_num, self.get_obs_size(is_opponent)),
                        'global_state': (self.get_state_size(is_opponent), ),
                        'action_mask': (agent_num, *self.action_helper.info().shape),
                    },
                    None,
                )
        return obs_space


    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def reward_space(self):
        return self._reward_space

    def __repr__(self):
        return "DI-engine SMAC Env"


def _flatten(obs, get_keys):
    new_obs = list()
    for agent_obs in obs:
        keys = get_keys(agent_obs)
        new_agent_obs = np.concatenate([agent_obs[feat_key].flatten() for feat_key in keys])
        new_obs.append(new_agent_obs)
    return new_obs


SMACTimestep = SMACEnv.SMACTimestep
SMACEnvInfo = SMACEnv.SMACEnvInfo
