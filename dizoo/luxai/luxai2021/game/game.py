import os
from luxai2021.game.replay import Replay
import math
import random
import sys
import traceback

from luxai2021.game.actions import MoveAction, PillageAction, SpawnCartAction, SpawnCityAction, SpawnWorkerAction, \
    ResearchAction, TransferAction
from .city import City
from .position import Position
from .constants import Constants, LuxMatchConfigs_Default
from .game_map import GameMap
from .unit import Worker, Cart

INPUT_CONSTANTS = Constants.INPUT_CONSTANTS
DIRECTIONS = Constants.DIRECTIONS

class MatchWarn(Exception):
    pass

class Game:
    def __init__(self, configs=None, agents=[]):
        """

        :param configs:
        :param agents:
        """
        # Initializations from src/Game/index.ts -> Game()
        self.configs = dict(LuxMatchConfigs_Default) # Shallow copy
        self.configs.update(configs)  # Override default config from specified config
        self.agents = []
        self.stop_replay_logging()
        self.reset()
        self.log_file = None
        

    def start_replay_logging(self, stateful=False, replay_folder="./replays/", replay_filename_prefix="replay"):
        """
        If replay_folder is not None, it enables saving of replays for every game into
        the target folder. Naming of each of the game replays is by appending a random number, eg:
            ./replays/replay_<random num>.json
            ./replays/replay_<random num>.json
            ...

        Args:
            replay_folder (str, optional): [description]. Defaults to "/replays/".
            replay_filename_prefix: Prefix to the filenames for the replay.
        """

        # Replays only work if a map seed is specified
        assert "seed" in self.configs, "Replays only work when a seed is specified."
        assert self.configs["seed"] is not None, "Replays only work when a seed is specified."

        # Create target folder if needed
        if not os.path.exists(replay_folder):
            os.makedirs(replay_folder)

        # Decide on the target file
        filename = f"{replay_filename_prefix}_{random.randint(0,10000)}.json"

        self.replay = Replay( self, os.path.join(replay_folder, filename), stateful )
        self.replay_stateful = stateful
        self.replay_folder = replay_folder
        self.replay_filename_prefix = replay_filename_prefix
    
    def stop_replay_logging(self):
        """
        Disables saving of replays at the end of each game.
        """
        self.replay = None
        self.replay_stateful = None
        self.replay_folder = None
        self.replay_filename_prefix = None

    def reset(self, updates=None, increment_turn=False):
        """
        Resets the game for another game.
        Updates are optionally a list of command messages from the kaggle controller
        that defines the state of the game to reset the game to. This gets sent from
        the kaggle server to our engine each turn.
        :param updates:
        :param increment_turn: Prevents resettig of turn count, and increments it by 1.
        """
        self.global_city_id_count = 0
        self.global_unit_id_count = 0
        self.cities = {}  # string -> City
        self.cells_with_roads = set() # Set, maintained to speed up agent designs that want to build road maps
        self.stats = {
            "teamStats": {
                Constants.TEAM.A: {
                    "fuelGenerated": 0,
                    "resourcesCollected": {
                        "wood": 0,
                        "coal": 0,
                        "uranium": 0,
                    },
                    "cityTilesBuilt": 0,
                    "workersBuilt": 0,
                    "cartsBuilt": 0,
                    "roadsBuilt": 0,
                    "roadsPillaged": 0,
                },
                Constants.TEAM.B: {
                    "fuelGenerated": 0,
                    "resourcesCollected": {
                        "wood": 0,
                        "coal": 0,
                        "uranium": 0,
                    },
                    "cityTilesBuilt": 0,
                    "workersBuilt": 0,
                    "cartsBuilt": 0,
                    "roadsBuilt": 0,
                    "roadsPillaged": 0,
                },
            },
        }

        # Option to keep game state turn number, and increment it.
        turn = 0
        if increment_turn:
            turn = self.state["turn"] + 1

        self.state = {
            "turn": turn,
            "teamStates": {
                Constants.TEAM.A: {
                    "researchPoints": 0,
                    "units": {},
                    "researched": {
                        "wood": True,
                        "coal": False,
                        "uranium": False,
                    }
                },
                Constants.TEAM.B: {
                    "researchPoints": 0,
                    "units": {},
                    "researched": {
                        "wood": True,
                        "coal": False,
                        "uranium": False,
                    }
                },
            }
        }

        # Generate the map
        self.map = GameMap(self.configs)
        self.map.generate_map(self)

        if self.replay:
            # Clear the replay log without writing
            self.replay.clear(self)

        self.process_updates(updates)

    def process_updates(self, updates, assign=True):

        if updates is None:
            return
        
        # Loop through updating the game from the list of updates
        # Implements /kits/python/simple/lux/game.py -> _update()
        for update in updates:
            if update == "D_DONE":
                break
            strings = update.split(" ")

            input_identifier = strings[0]
            if input_identifier == INPUT_CONSTANTS.RESEARCH_POINTS:
                team = int(strings[1])
                research_points = int(strings[2])
                if assign:
                    self.state["teamStates"][team]["researchPoints"] = research_points
                else:
                    assert self.state["teamStates"][team]["researchPoints"] == research_points

                if int(strings[2]) >= self.configs["parameters"]["RESEARCH_REQUIREMENTS"]["COAL"]:
                    if assign:
                        self.state["teamStates"][team]["researched"]["coal"] = True
                    else:
                        assert self.state["teamStates"][team]["researched"]["coal"] == True

                if int(strings[2]) >= self.configs["parameters"]["RESEARCH_REQUIREMENTS"]["URANIUM"]:
                    if assign:
                        self.state["teamStates"][team]["researched"]["uranium"] = True
                    else:
                        assert self.state["teamStates"][team]["researched"]["uranium"] == True

            elif input_identifier == INPUT_CONSTANTS.RESOURCES:
                r_type = strings[1]
                x = int(strings[2])
                y = int(strings[3])
                amt = int(float(strings[4]))
                if assign:
                    self.map.add_resource(x, y, r_type, amt)
                else:
                    cell = self.map.get_cell(x, y)
                    assert cell.resource.amount == amt
                    assert cell.resource.type == r_type 

            elif input_identifier == INPUT_CONSTANTS.UNITS:
                unit_type = int(strings[1])
                team = int(strings[2])
                unit_id = strings[3]
                x = int(strings[4])
                y = int(strings[5])
                cooldown = float(strings[6])
                wood = int(strings[7])
                coal = int(strings[8])
                uranium = int(strings[9])
                if assign:
                    if unit_type == Constants.UNIT_TYPES.WORKER:
                        self.spawn_worker(team, x, y, unit_id, cooldown=cooldown,
                                        cargo={"wood": wood, "uranium": uranium, "coal": coal})
                    elif unit_type == Constants.UNIT_TYPES.CART:
                        self.spawn_cart(team, x, y, unit_id, cooldown=cooldown,
                                        cargo={"wood": wood, "uranium": uranium, "coal": coal})
                else:
                    cell = self.map.get_cell(x, y)
                    assert len(cell.units) > 0
                    assert unit_id in [u.id for u in cell.units.values()], f'unit id {unit_id} missplaced'

            elif input_identifier == INPUT_CONSTANTS.CITY:
                team = int(strings[1])
                city_id = strings[2]
                fuel = float(strings[3])
                light_upkeep = float(strings[4])  # Unused
                if assign:
                    self.cities[city_id] = City(team, self.configs, None, city_id, fuel)
                else:
                    assert city_id in self.cities

            elif input_identifier == INPUT_CONSTANTS.CITY_TILES:
                team = int(strings[1])
                city_id = strings[2]
                x = int(strings[3])
                y = int(strings[4])
                cooldown = float(strings[5])
                city = self.cities[city_id]
                cell = self.map.get_cell(x, y)
                if assign:
                    cell.set_city_tile(team, city_id, cooldown)
                    city.add_city_tile(cell)
                    self.stats["teamStats"][team]["cityTilesBuilt"] += 1
                else:
                    assert cell.city_tile.city_id == city_id
                    assert cell in city.city_cells

            elif input_identifier == INPUT_CONSTANTS.ROADS:
                x = int(strings[1])
                y = int(strings[2])
                road = float(strings[3])
                cell = self.map.get_cell(x, y)
                if cell not in self.cells_with_roads:
                    self.cells_with_roads.add(cell)
                if assign:
                    cell.road = road
                else:
                    assert cell.get_road() == road

    def _gen_initial_accumulated_action_stats(self):
        """
        Initial stats
        Implements src/Game/index.ts -> Game._genInitialAccumulatedActionStats()
        :return:
        """
        return {
            Constants.TEAM.A: {
                "workersBuilt": 0,
                "cartsBuilt": 0,
                "actionsPlaced": set(),
            },
            Constants.TEAM.B: {
                "workersBuilt": 0,
                "cartsBuilt": 0,
                "actionsPlaced": set(),
            },
        }

    def action_from_command(self, cmd):
        """
        Converts a match text command to an action. Validation is handled elsewhere.
        This is used in Kaggle submissions to decode actions taken and update the game.
        Somewhat implements src/Game/index.ts -> Game.validateCommand()
        :param cmd:
        """
        return self.action_from_command_low(cmd.command, cmd.agentID)

    def action_from_string(self, comm, agentID):
        try:
            return self.action_from_command_low(comm, agentID)
        except KeyError:
            print(f'action failed, probably a dead unit {agentID}: {comm}')
            return None

    def action_from_command_low(self, comm, agentID):

        invalid_msg = f"Agent {agentID} sent invalid command"
        malformed_msg = f"Agent {agentID} sent malformed command: {comm}"

        def check(condition, error_msg, trace=True):
            if condition:
                if trace:
                    raise Exception(error_msg + f"; turn {self.state['turn']}; cmd: {comm}")
                else:
                    raise Exception(error_msg)

        # Tokenize command
        parts = comm.split(' ')
        check(len(parts) <= 1, invalid_msg)
        action = parts[0]
        parts = parts[1:]

        # Load the team details
        team = agentID

        # Construct the action
        result = None
        if action == Constants.ACTIONS.PILLAGE:
            check(len(parts) != 1, malformed_msg)
            uid = parts[0]

            unit = self.get_unit(team, uid)
            check(unit is None, invalid_msg)
            result = PillageAction(team, uid)

        elif action == Constants.ACTIONS.BUILD_CITY:
            check(len(parts) != 1, malformed_msg)
            uid = parts[0]

            unit = self.get_unit(team, uid)
            check(unit is None, invalid_msg)

            result = SpawnCityAction(team, uid)

        elif action == Constants.ACTIONS.BUILD_CART:
            check(len(parts) != 2, malformed_msg)

            x = int(parts[0])
            y = int(parts[1])

            result = SpawnCartAction(team, None, x, y)

        elif action == Constants.ACTIONS.BUILD_WORKER:
            check(len(parts) != 2, malformed_msg)

            x = int(parts[0])
            y = int(parts[1])
            result = SpawnWorkerAction(team, None, x, y)

        elif action == Constants.ACTIONS.MOVE:
            check(len(parts) != 2, malformed_msg)

            uid = parts[0]
            direction = parts[1]

            result = MoveAction(team, uid, direction)

        elif action == Constants.ACTIONS.RESEARCH:
            check(len(parts) != 2, malformed_msg)

            x = int(parts[0])
            y = int(parts[1])

            result = ResearchAction(team, x, y, None)

        elif action == Constants.ACTIONS.TRANSFER:
            check(len(parts) != 4, malformed_msg)

            source_id = parts[0]
            destination_id = parts[1]
            resource_type = parts[2]
            amount = int(parts[3])

            result = TransferAction(
                team,
                source_id,
                destination_id,
                resource_type,
                amount
            )
        else:
            raise Exception(f"unknown action {action}")

        return result

    def run_turn_with_actions(self, actions):
        """
        Runs a single game turn with the specified actions
        Returns:
            True if game is still running
            False if game is over
        """
        if "log" in self.configs and self.configs["log"]:
            self.log('Processing turn ' + str(self.state["turn"]))

        if self.replay:
            # Log actions to a replay
            self.replay.add_actions(self, actions)
            
            

        # Loop over commands and validate and map into internal action representations
        actions_map = {}

        accumulated_action_stats = self._gen_initial_accumulated_action_stats()
        for i, action in enumerate(actions):
            # get the command and the agent that issued it and handle appropriately
            try:
                action = self.validate_command(
                    actions[i],
                    accumulated_action_stats
                )
                if action is not None:
                    if action.action in actions_map:
                        actions_map[action.action].append(action)
                    else:
                        actions_map[action.action] = [action]
            except Exception as e:
                self.log("Error processing action")
                self.log(repr(e))
                self.log(''.join(traceback.format_exception(None, e, e.__traceback__)))

        # give units and city tiles their validated actions to use
        if Constants.ACTIONS.BUILD_CITY in actions_map:
            for action in actions_map[Constants.ACTIONS.BUILD_CITY]:
                self.get_unit(action.team, action.unit_id).give_action(action)

        if Constants.ACTIONS.BUILD_WORKER in actions_map:
            for action in actions_map[Constants.ACTIONS.BUILD_WORKER]:
                city_tile = self.map.get_cell(action.x, action.y).city_tile
                city_tile.give_action(action)

        if Constants.ACTIONS.BUILD_CART in actions_map:
            for action in actions_map[Constants.ACTIONS.BUILD_CART]:
                city_tile = self.map.get_cell(action.x, action.y).city_tile
                city_tile.give_action(action)

        if Constants.ACTIONS.PILLAGE in actions_map:
            for action in actions_map[Constants.ACTIONS.PILLAGE]:
                self.get_unit(action.team, action.unit_id).give_action(action)

        if Constants.ACTIONS.RESEARCH in actions_map:
            for action in actions_map[Constants.ACTIONS.RESEARCH]:
                city_tile = self.map.get_cell(action.x, action.y).city_tile
                city_tile.give_action(action)

        if Constants.ACTIONS.TRANSFER in actions_map:
            for action in actions_map[Constants.ACTIONS.TRANSFER]:
                self.get_unit(action.team, action.source_id).give_action(action)

        if Constants.ACTIONS.MOVE in actions_map:
            pruned_move_actions = self.handle_movement_actions(
                actions_map[Constants.ACTIONS.MOVE]
            )
        else:
            pruned_move_actions = []

        for action in pruned_move_actions:
            # if direction is center, ignore it
            if action.direction != Constants.DIRECTIONS.CENTER:
                self.get_unit(action.team, action.unit_id).give_action(action)

        # now we go through every actionable entity and execute actions
        for city in self.cities.values():
            for city_cell in city.city_cells:
                try:
                    city_cell.city_tile.handle_turn(self)
                except Exception as e:
                    self.log("Critical error handling city turn.")
                    self.log(repr(e))
                    self.log(''.join(traceback.format_exception(None, e, e.__traceback__)))

        teams = [Constants.TEAM.A, Constants.TEAM.B]
        for team in teams:
            for unit in self.state["teamStates"][team]["units"].values():
                try:
                    unit.handle_turn(self)
                except Exception as e:
                    self.log("Critical error handling unit turn.")
                    self.log(repr(e))
                    self.log(''.join(traceback.format_exception(None, e, e.__traceback__)))

        # distribute all resources in order of decreasing fuel efficiency
        self.distribute_all_resources()

        # now we make all units with cargo drop all resources on the city they are standing on
        for team in teams:
            for unit in self.state["teamStates"][team]["units"].values():
                self.handle_resource_deposit(unit)

        if self.is_night():
            self.handle_night()

        # remove resources that are depleted from map
        new_resources_map = []
        self.map.resources_by_type = {}
        for i in range(len(self.map.resources)):
            cell = self.map.resources[i]
            if cell.resource.amount > 0:
                new_resources_map.append(cell)
                if cell.resource.type not in self.map.resources_by_type:
                    self.map.resources_by_type[cell.resource.type] = [cell]
                else:
                    self.map.resources_by_type[cell.resource.type].append(cell)

        self.map.resources = new_resources_map

        # regenerate forests
        self.regenerate_trees()

        match_over = self.match_over()

        self.state["turn"] += 1

        # store state for replays
        if self.replay:
            self.replay.add_state(self)

        self.run_cooldowns()

        if match_over:
            if self.replay:
                # Write the replay to a file
                self.replay.write(self)

                # Start a new replay file for the next game
                self.start_replay_logging(self.replay_stateful, self.replay_folder, self.replay_filename_prefix)
            return True

        # self.log('Beginning turn %s' % self.state["turn"])
        return False

    def handle_night(self):
        """
        Implements /src/logic.ts -> handleNight()
        /**
        * Handle nightfall and update state accordingly
        */
        """
        for city in list(self.cities.values()):
            # if city does not have enough fuel, destroy it
            # TODO, probably add this event to replay
            if city.fuel < city.get_light_upkeep():
                self.destroy_city(city.team, city.id)
            else:
                city.fuel -= city.get_light_upkeep()

        for team in [Constants.TEAM.A, Constants.TEAM.B]:
            for unit in list(self.state["teamStates"][team]["units"].values()):
                # TODO: add condition for different light upkeep for units stacked on a city.
                if not self.map.get_cell_by_pos(unit.pos).is_city_tile():
                    if not unit.spend_fuel_to_survive():
                        # delete unit
                        self.destroy_unit(unit.team, unit.id)

    def run_cooldowns(self):
        """
        Implements /src/Game/index.ts -> runCooldowns()
        """
        for team in [Constants.TEAM.A, Constants.TEAM.B]:
            units = self.get_teams_units(team).values()
            for unit in units:
                unit.cooldown -= self.map.get_cell_by_pos(unit.pos).get_road()
                unit.cooldown = max(unit.cooldown - 1, 0)

    def match_over(self):
        """
        Implements /src/logic.ts -> matchOver()
        /**
        * Determine if match is over or not
        */
        """

        if self.state["turn"] >= self.configs["parameters"]["MAX_DAYS"] - 1:
            return True

        # over if at least one team has no units left or city tiles
        teams = [Constants.TEAM.A, Constants.TEAM.B]
        city_count = [0, 0]

        for city in self.cities.values():
            city_count[city.team] += 1

        for team in teams:
            if len(self.get_teams_units(team)) + city_count[team] == 0:
                return True

        return False

    def get_winning_team(self):
        """
        Implements /src/logic.ts -> getResults()
        """

        # count city tiles
        city_tile_count = [0, 0]
        for city in self.cities.values():
            city_tile_count[city.team] += len(city.city_cells)

        if city_tile_count[Constants.TEAM.A] > city_tile_count[Constants.TEAM.B]:
            return Constants.TEAM.A
        elif city_tile_count[Constants.TEAM.A] < city_tile_count[Constants.TEAM.B]:
            return Constants.TEAM.B

        # if tied, count by units
        unit_count = [
            len(self.get_teams_units(Constants.TEAM.A)),
            len(self.get_teams_units(Constants.TEAM.B)),
        ]
        if unit_count[Constants.TEAM.A] > unit_count[Constants.TEAM.B]:
            return Constants.TEAM.A
        elif unit_count[Constants.TEAM.B] > unit_count[Constants.TEAM.A]:
            return Constants.TEAM.B

        # if tied still, count by fuel generation
        if (
                self.stats["teamStats"][Constants.TEAM.A]["fuelGenerated"] >
                self.stats["teamStats"][Constants.TEAM.B]["fuelGenerated"]
        ):
            return Constants.TEAM.A
        elif (
                self.stats["teamStats"][Constants.TEAM.A]["fuelGenerated"] <
                self.stats["teamStats"][Constants.TEAM.B]["fuelGenerated"]
        ):
            return Constants.TEAM.B

        # if still undecided, for now, go by random choice
        if random.random() > 0.5:
            return Constants.TEAM.A
        return Constants.TEAM.B

    def log(self, text):
        """
        Logs the specified text
        :param text:
        """
        if self.log_file is None:
            self.log_file = open("log.txt", "w")
        if text is not None:
            self.log_file.write(text + "\n")

    def validate_command(self, cmd, accumulated_action_stats=None):
        """
        Returns an Action object if validated. If invalid, throws MatchWarn
        Implements src/Game/index.ts -> Game.validateCommand()
        """
        if accumulated_action_stats is None:
            accumulated_action_stats = self._gen_initial_accumulated_action_stats()

        acc = accumulated_action_stats[cmd.team]
        # TODO: IMPLEMENT THIS
        if isinstance(cmd, SpawnCityAction):
            unit = self.get_unit(cmd.team, cmd.unit_id)
            if unit is None:
                raise MatchWarn("Agent tried to build CityTile with invalid/unowned unit id: {}".format(cmd.unit_id))
            cell = self.map.get_cell_by_pos(unit.pos)
            
            if cell.is_city_tile():
                raise MatchWarn("Agent tried to build CityTile on existing CityTile")
                
            if cell.has_resource():
                raise MatchWarn("Agent tried to build CityTile on non-empty resource tile")
            if not unit.can_act():
                raise MatchWarn("Agent tried to build CityTile with cooldown: {}".format(unit.cooldown))
                
            cargoTotal = unit.cargo['wood'] + unit.cargo['coal']+ unit.cargo['uranium']
            
            if cargoTotal < self.configs['parameters']['CITY_BUILD_COST']:
                raise MatchWarn("Agent tried to build CityTile with insufficient materials wood + coal + uranium: {}".format(cargoTotal))
            acc['actionsPlaced'].add(cmd.unit_id)
            return cmd
        elif isinstance(cmd, MoveAction):
            unit = self.get_unit(cmd.team, cmd.unit_id)
            if unit is None:
                raise MatchWarn("Agent tried to move unit {} that it does not own".format(cmd.unit_id))
            if not unit.can_move():
                raise MatchWarn("Agent tried to move unit {} with cooldown: {}".format(cmd.unit_id, unit.cooldown))
            if not cmd.direction in [ Constants.DIRECTIONS.CENTER,
                                     Constants.DIRECTIONS.EAST,
                                     Constants.DIRECTIONS.NORTH,
                                     Constants.DIRECTIONS.SOUTH,
                                     Constants.DIRECTIONS.WEST ]:
                raise MatchWarn("Agent tried to move unit {} in invalid direction {}".format(cmd.unit_id, cmd.direction))
            if cmd.direction != Constants.DIRECTIONS.CENTER:
                new_pos = unit.pos.translate(cmd.direction, 1)
                if not self.map.in_map(new_pos):
                    raise MatchWarn("Agent tried to move unit {} off map".format(cmd.unit_id))
                if self.map.get_cell_by_pos(new_pos).is_city_tile() and self.map.get_cell_by_pos(new_pos).city_tile.team != cmd.team:
                        raise MatchWarn("Agent tried to move unit {} onto opponent CityTile".format(cmd.unit_id))
            acc['actionsPlaced'].add(cmd.unit_id)
            return cmd
        elif isinstance(cmd, SpawnWorkerAction) or isinstance(cmd, SpawnCartAction):
            if not self.map.in_map(Position(cmd.x, cmd.y)):
                raise MatchWarn("Agent tried to build unit with invalid coordinates")
            cell = self.map.get_cell(cmd.x, cmd.y)
            if (not cell.is_city_tile()) ^ (cell.city_tile.team != cmd.team):
                raise MatchWarn("Agent tried to build unit on tile ({}, {}) that it does not own".format(cmd.x, cmd.y))
            city_tile = cell.city_tile
            if not city_tile.can_build_unit():
                raise MatchWarn("Agent tried to build unit on tile ({}, {}) but CityTile still with cooldown of {}".format(cmd.x, cmd.y, city_tile.cooldown))
            if isinstance(cmd, SpawnCartAction):
                if self.cart_unit_cap_reached(cmd.team, acc['cartsBuilt'] + acc['workersBuilt']):
                    raise MatchWarn("Agent tried to build cart on tile ({}, {}) but unit cap reached. Build more CityTiles!".format(cmd.x, cmd.y))
            else: # SpawnWorkerAction
                if self.worker_unit_cap_reached(cmd.team, acc['cartsBuilt'] + acc['workersBuilt']):
                    raise MatchWarn("Agent tried to build worker on tile ({}, {}) but unit cap reached. Build more CityTiles!".format(cmd.x, cmd.y))
            
            acc['actionsPlaced'].add(city_tile.get_tile_id())
            if isinstance(cmd, SpawnCartAction):
                acc['cartsBuilt'] += 1
                return cmd
            else:
                acc['workersBuilt'] += 1
                return cmd
        
            
        else:        
            # no check. bad.
            return cmd

    def worker_unit_cap_reached(self, team, offset=0):
        """
        Returns True if unit cap reached
        Implements src/Game/index.ts -> Game.workerUnitCapReached()
        """
        team_city_tile_count = 0
        for city in self.cities.values():
            if city.team == team:
                team_city_tile_count += len(city.city_cells)
        
        return len(self.state["teamStates"][team]["units"]) + offset >= team_city_tile_count

    def cart_unit_cap_reached(self, team, offset=0):
        """
        Returns True if unit cap reached
        Implements src/Game/index.ts -> Game.cartUnitCapReached()
        """
        return self.worker_unit_cap_reached(team, offset)

    def spawn_worker(self, team, x, y, unit_id=None, cooldown=0.0, cargo=None):
        """
        Spawns new worker
        Implements src/Game/index.ts -> Game.spawnWorker()
        """
        if cargo is None:
            cargo = {"wood": 0, "uranium": 0, "coal": 0}
        cell = self.map.get_cell(x, y)
        unit = Worker(
            x,
            y,
            team,
            self.configs,
            self.global_unit_id_count + 1,
            cooldown,
            cargo
        )

        if unit_id:
            unit.id = unit_id
        else:
            self.global_unit_id_count += 1

        cell.units[unit.id] = unit

        self.state["teamStates"][team]["units"][unit.id] = unit
        self.stats["teamStats"][team]["workersBuilt"] += 1
        return unit

    def spawn_cart(self, team, x, y, unit_id=None, cooldown=0.0, cargo=None):
        """
        Spawns new cart
        Implements src/Game/index.ts -> Game.spawnCart()
        """
        if cargo is None:
            cargo = {"wood": 0, "uranium": 0, "coal": 0}
        cell = self.map.get_cell(x, y)
        unit = Cart(x,
                    y,
                    team,
                    self.configs,
                    self.global_unit_id_count + 1,
                    cooldown,
                    cargo)
        if unit_id:
            unit.id = unit_id
        else:
            self.global_unit_id_count += 1

        cell.units[unit.id] = unit
        self.state["teamStates"][team]["units"][unit.id] = unit
        self.stats["teamStats"][team]["cartsBuilt"] += 1
        return unit

    def spawn_city_tile(self, team, x, y, city_id=None):
        """
        Spawns new city tile
        Implements src/Game/index.ts -> Game.spawnCityTile()
        """
        cell = self.map.get_cell(x, y)

        # now update the cities field accordingly
        adj_cells = self.map.get_adjacent_cells(cell)

        city_ids_found = []

        adj_same_team_city_tiles = []
        for cell2 in adj_cells:
            if cell2.is_city_tile() and cell2.city_tile.team == team:
                adj_same_team_city_tiles.append(cell2)
                if cell2.city_tile.city_id not in city_ids_found:
                    city_ids_found.append(cell2.city_tile.city_id)

        # if no adjacent city cells of same team, generate new city
        if len(adj_same_team_city_tiles) == 0:
            city = City(team, self.configs, self.global_city_id_count + 1)

            if city_id is not None:
                city.id = city_id
            else:
                self.global_city_id_count += 1

            cell.set_city_tile(team, city.id)
            city.add_city_tile(cell)
            self.cities[city.id] = city
            return cell.city_tile

        else:
            # otherwise add tile to city
            city_id = adj_same_team_city_tiles[0].city_tile.city_id
            city = self.cities[city_id]
            cell.set_city_tile(team, city_id)

            # update adjacency counts for bonuses
            cell.city_tile.adjacent_city_tiles = len(adj_same_team_city_tiles)
            for adjCell in adj_same_team_city_tiles:
                adjCell.city_tile.adjacent_city_tiles += 1
            city.add_city_tile(cell)

            # update all merged cities' cells with merged city_id, move to merged city and delete old city
            for local_id in city_ids_found:
                if local_id != city_id:
                    old_city = self.cities[local_id]
                    for cell3 in old_city.city_cells:
                        cell3.city_tile.city_id = city_id
                        city.add_city_tile(cell3)

                    city.fuel += old_city.fuel
                    self.cities.pop(old_city.id)

            return cell.city_tile

    def move_unit(self, team, unit_id, direction):
        """
        Moves a unit
        Implements src/Game/index.ts -> Game.moveUnit()
        """
        unit = self.get_unit(team, unit_id)

        # remove unit from old cell and move to new one and update unit pos
        self.map.get_cell_by_pos(unit.pos).units.pop(unit.id)
        unit.pos = unit.pos.translate(direction, 1)
        self.map.get_cell_by_pos(unit.pos).units[unit.id] = unit

    def distribute_all_resources(self):
        """
        Distributes resources
        Implements src/Game/index.ts -> Game.distributeAllResources()
        """
        mining_order = [
            Constants.RESOURCE_TYPES.URANIUM,
            Constants.RESOURCE_TYPES.COAL,
            Constants.RESOURCE_TYPES.WOOD,
        ]

        for curType in mining_order:
            self.handle_resource_type_release(curType)

    def handle_resource_type_release(self, resource_type):
        """
        * For each unit, check current and orthoganally adjacent cells for that resource
        * type. If found, request as much as we can carry from these cells. In the case of un-even 
        * amounts, the unit will request an equal amount from all tiles to fill their cargo, then
        * discard the rest. (for example on 3 wood tiles with 60 wood it would request 17 to each
        * wood tile and discard/waste the extra 1 wood mined).
        * 
        * If the unit is on a city tile, only one request will be made (even if there are 
        * multiple workers on the tile )and the resources will be deposited into the city as fuel.
        * 
        * Once all units have requested resources, distribute the resources, reducing requests
        * requests if it would exceed the current value. In this case the remaining
        * will be distributed evenly, with the leftovers discarded.
        * 
        * @param resourceType - the type of the resource
        * Description copy pasted from src/Game/index.ts -> Game.handleResourceTypeRelease()
        """
        # build up the resource requests
        requests = self.create_resource_requests(resource_type)

        # resolve resource requests
        self.resolve_resource_requests(resource_type, requests)

    class ResourceRequest:
        def __init__(self, from_pos, amount, worker, city):
            """

            :param from_pos:
            :param amount:
            :param worker:
            :param city:
            """
            self.fromPos = from_pos
            self.amount = amount
            self.worker = worker
            self.city = city

        def __eq__(self, other) -> bool:
            """

            :param other:
            :return:
            """
            return (
                    self.fromPos == other.fromPos
                    and (self.worker.id if self.worker else None) == (other.worker.id if other.worker else None)
                    and self.amount == other.amount
                    and (self.city.id if self.city else None) == (other.city.id if other.city else None)
            )

    def create_resource_requests(self, resource_type):
        """

        :param resource_type:
        :return:
        """
        type_map = {
            Constants.RESOURCE_TYPES.WOOD: "WOOD",
            Constants.RESOURCE_TYPES.COAL: "COAL",
            Constants.RESOURCE_TYPES.URANIUM: "URANIUM",
        }
        mining_rate = rate = self.configs["parameters"]["WORKER_COLLECTION_RATE"][type_map[resource_type]]
        reqs = {}
        for team in [Constants.TEAM.A, Constants.TEAM.B]:
            if self.state["teamStates"][team]["researched"][resource_type]:
                for unit in self.state["teamStates"][team]["units"].values():
                    if unit.type == Constants.UNIT_TYPES.WORKER:
                        unit_cell = self.map.get_cell_by_pos(unit.pos)
                        cells = [unit_cell] + self.map.get_adjacent_cells(unit_cell)
                        minable = [c for c in cells if c.has_resource() and c.resource.type == resource_type]
                        if len(minable) > 0:
                            mine_amount = min(math.ceil(unit.get_cargo_space_left() / len(minable)), mining_rate)
                        for cell in minable:
                            if cell.pos not in reqs:
                                reqs[cell.pos] = []
                            req = Game.ResourceRequest(
                                unit.pos,
                                mine_amount,
                                None if unit_cell.is_city_tile() else unit,
                                self.cities[unit_cell.city_tile.city_id] if unit_cell.is_city_tile() else None,
                            )
                            has_req = req in reqs[cell.pos]
                            if not has_req:
                                reqs[cell.pos].append(req)
        return reqs

    def resolve_resource_requests(self, resource_type, requests):
        """

        :param resource_type:
        :param requests:
        :return:
        """
        type_map = {
            Constants.RESOURCE_TYPES.WOOD: "WOOD",
            Constants.RESOURCE_TYPES.COAL: "COAL",
            Constants.RESOURCE_TYPES.URANIUM: "URANIUM",
        }
        conversion_rate = self.configs["parameters"]["RESOURCE_TO_FUEL_RATE"][type_map[resource_type]]
        for position, reqs in requests.items():
            amount_left = self.map.get_cell_by_pos(position).resource.amount
            amounts_reqs = reqs  # dont make tuples like typescript reference implementation
            while len(amounts_reqs) > 0 and sum([req.amount for req in amounts_reqs]) > 0 and amount_left > 0:
                to_fill = min(min([req.amount for req in amounts_reqs]), math.floor(amount_left / len(amounts_reqs)))
                for r in amounts_reqs:
                    if r.city is not None:
                        self.stats["teamStats"][r.city.team]["resourcesCollected"][resource_type] += to_fill
                        r.city.fuel += to_fill * conversion_rate
                    else:
                        to_give = min(r.worker.get_cargo_space_left(), to_fill)
                        self.stats["teamStats"][r.worker.team]["resourcesCollected"][resource_type] += to_give
                        r.worker.cargo[resource_type] += to_give
                    r.amount -= to_fill
                amount_left -= to_fill * len(amounts_reqs)
                if amount_left < len(amounts_reqs):
                    amount_left = 0
                amounts_reqs = list(filter(lambda r: r.amount > 0, amounts_reqs))
            cell = self.map.get_cell_by_pos(position)
            cell.resource.amount = amount_left

    def handle_resource_deposit(self, unit):
        """
        Auto deposit resources of unit to tile it is on
        Implements src/Game/index.ts -> Game.handleResourceDeposit()
        """
        cell = self.map.get_cell_by_pos(unit.pos)
        if cell.is_city_tile() and cell.city_tile.team == unit.team:
            city = self.cities.get(cell.city_tile.city_id)
            fuel_gained = 0
            fuel_gained += unit.cargo["wood"] * self.configs["parameters"]["RESOURCE_TO_FUEL_RATE"]["WOOD"]
            fuel_gained += unit.cargo["coal"] * self.configs["parameters"]["RESOURCE_TO_FUEL_RATE"]["COAL"]
            fuel_gained += unit.cargo["uranium"] * self.configs["parameters"]["RESOURCE_TO_FUEL_RATE"]["URANIUM"]
            city.fuel += fuel_gained

            self.stats["teamStats"][unit.team]["fuelGenerated"] += fuel_gained

            unit.cargo = {
                "wood": 0,
                "uranium": 0,
                "coal": 0,
            }

    def get_teams_units(self, team):
        """
        Get list of units.
        Implements src/Game/index.ts -> Game.getTeamsUnits()
        """
        return self.state["teamStates"][team]["units"]

    def get_unit(self, team, unit_id):
        """
        Get the specific unit.
        Implements src/Game/index.ts -> Game.getUnit()
        """
        return self.state["teamStates"][team]["units"][unit_id]

    def transfer_resources(self, team, source_id, destination_id, resource_type, amount):
        """
        Transfer resources on a given team between 2 units. This does not check adjacency requirement, but its expected
        that the 2 units are adjacent. This allows for simultaneous movement of 1 unit and transfer of another
        Implements src/Game/index.ts -> transferResources()
        """
        source_unit = self.get_unit(team, source_id)
        destination_unit = self.get_unit(team, destination_id)
        # the amount to actually transfer is the minimum of:
        transfer_amount = min(
            # the amount requested
            amount,
            # and all that we have if that's less than requested
            source_unit.cargo[resource_type],
            # and no more than destination-unit's remaining cargo-space
            destination_unit.get_cargo_space_left()
        )
        source_unit.cargo[resource_type] -= transfer_amount
        destination_unit.cargo[resource_type] += transfer_amount

    def destroy_city(self, team, city_id):
        """
        Destroys the unit with this id and team and removes from tile
        Implements src/Game/index.ts -> Game.destroyCity()
        """
        city = self.cities.get(city_id)
        self.cities.pop(city_id)
        for cell in city.city_cells:
            cell.city_tile = None
            cell.road = self.configs["parameters"]["MIN_ROAD"]
            if cell in self.cells_with_roads:
                self.cells_with_roads.remove(cell)

    def destroy_unit(self, team, unit_id):
        """
        Destroys the unit with this id and team and removes from tile
        Implements src/Game/index.ts -> Game.destroyUnit()
        """
        unit = self.get_unit(team, unit_id)
        self.map.get_cell_by_pos(unit.pos).units.pop(unit_id)
        self.state["teamStates"][team]["units"].pop(unit_id)

    def regenerate_trees(self):
        """
        Regenerate trees
        Implements src/Game/index.ts -> Game.regenerateTrees()
        /**
        * regenerates trees on map according to the following formula
        * let max_wood_amount be base and the current amount be curr
        *
        * then at the end of each turn after all moves and all resource collection is finished,
        * the wood at a wood tile grows to ceil(min(curr * 1.03, base))
        */
        """
        if Constants.RESOURCE_TYPES.WOOD in self.map.resources_by_type:
            for cell in self.map.resources_by_type[Constants.RESOURCE_TYPES.WOOD]:
                # add this condition so we let forests near a city start large (but not regrow until below a max)
                if cell.resource.amount < self.configs["parameters"]["MAX_WOOD_AMOUNT"]:
                    cell.resource.amount = math.ceil(
                        min(
                            cell.resource.amount * self.configs["parameters"]["WOOD_GROWTH_RATE"],
                            self.configs["parameters"]["MAX_WOOD_AMOUNT"]
                        )
                    )

    def handle_movement_actions(self, actions):
        """
        Process given move actions and returns a pruned array of actions that can all be executed with no collisions
        Implements src/Game/index.ts -> Game.handleMovementActions()
        /**
        * Algo:
        *
        * iterate through all moves and store a mapping from cell to the actions that will cause a unit to move there
        *
        * for each cell that has multiple mapped to actions, we remove all actions as that cell is a "bump" cell
        * where no units can get there because they all bumped into each other
        *
        * for all removed actions for that particular cell, find the cell the unit that wants to execute the action is
        * currently at, labeled `origcell`. Revert these removed actions by first getting all the actions mapped from
        * `origcell` and then deleting that mapping, and then recursively reverting the actions mapped from `origcell`
        *
        */
        """
        cells_to_actions_to_there = {}
        moving_units = set()

        for action in actions:
            new_cell = self.map.get_cell_by_pos(
                self.get_unit(action.team, action.unit_id).pos.translate(action.direction, 1)
            )
            if new_cell is not None:
                # new_cell = action.new_cell
                if new_cell in cells_to_actions_to_there:
                    cells_to_actions_to_there[new_cell] += [action]
                else:
                    cells_to_actions_to_there[new_cell] = [action]

                moving_units.add(action.unit_id)

        def revert_action(action):
            # reverts a given action such that cellsToActionsToThere has no collisions due to action and all related actions
            self.log(
                f"turn {{self.state['turn']}} Unit {{action.unit_id}} collided when trying to move {{action.direction}} to ({{action.newcell.pos.x}}, {{action.newcell.pos.y}})")

            original_cell = self.map.get_cell_by_pos(
                self.get_unit(action.team, action.unit_id).pos
            )

            # get the colliding actions caused by a revert of the given action and then delete them from the mapped origcell provided it is not a city tile
            colliding_actions = cells_to_actions_to_there[
                original_cell] if original_cell in cells_to_actions_to_there else None
            if not original_cell.is_city_tile():
                if colliding_actions is not None:
                    cells_to_actions_to_there.pop(original_cell)

                    # for each colliding action, revert it.
                    for collidingAction in colliding_actions:
                        revert_action(collidingAction)

        actioned_cells = list(cells_to_actions_to_there.keys())
        for cell in actioned_cells:
            if cell in cells_to_actions_to_there:
                current_actions = cells_to_actions_to_there[cell]
                actions_to_revert = []
                if current_actions is not None:
                    if len(current_actions) > 1:
                        # only revert actions that are going to the same tile that is not a city
                        # if going to the same city tile, we know those actions are from same team units, and is allowed
                        if not cell.is_city_tile():
                            actions_to_revert += current_actions
                    elif len(current_actions) == 1:
                        # if there is just one move action, check there isn't a unit on there that is not moving and not a city tile
                        action = current_actions[0]
                        if not cell.is_city_tile():
                            if len(cell.units) == 1:
                                unit_there_is_still = True
                                for unit in cell.units.values():
                                    if unit.id in moving_units:
                                        unit_there_is_still = False
                                if unit_there_is_still:
                                    actions_to_revert.append(action)

            # if there are collisions, revert those actions and remove the mapping
            for action in actions_to_revert:
                revert_action(action)
            for action in actions_to_revert:
                new_cell = self.map.get_cell_by_pos(
                    self.get_unit(action.team, action.unit_id).pos.translate(action.direction, 1)
                )
                if new_cell in cells_to_actions_to_there:
                    cells_to_actions_to_there.pop(new_cell)

        pruned_actions = []
        for current_actions in cells_to_actions_to_there.values():
            pruned_actions += current_actions

        return pruned_actions

    def is_night(self):
        """
        Is it night.
        Implements src/Game/index.ts -> Game.isNight()
        """
        day_length = self.configs["parameters"]["DAY_LENGTH"]
        cycle_length = day_length + self.configs["parameters"]["NIGHT_LENGTH"]
        return (self.state["turn"] % cycle_length) >= day_length

    def to_state_object(self):
        """
        Serialize state
        Implements src/Game/index.ts -> Game.toStateObject()
        """
        cities = {}
        for city in self.cities.values():
            city_cells = []
            for cell in city.city_cells:
                city_cells.append({
                    "x": cell.pos.x,
                    "y": cell.pos.y,
                    "cooldown": cell.city_tile.cooldown,
                })

            cities[city.id] = {
                "id": city.id,
                "fuel": city.fuel,
                "lightupkeep": city.get_light_upkeep(),
                "team": city.team,
                "cityCells": city_cells
            }

        state = {
            "turn": self.state["turn"],
            "globalCityIDCount": self.global_city_id_count,
            "globalunit_idCount": self.global_unit_id_count,
            "teamStates": {
                Constants.TEAM.A: {
                    "researchPoints": 0,
                    "units": {},
                    "researched": {
                        "wood": True,
                        "coal": False,
                        "uranium": False,
                    },
                },
                Constants.TEAM.B: {
                    "researchPoints": 0,
                    "units": {},
                    "researched": {
                        "wood": True,
                        "coal": False,
                        "uranium": False,
                    },
                },
            },
            "map": self.map.to_state_object(),
            "cities": cities,
        }

        teams = [Constants.TEAM.A, Constants.TEAM.B]
        for team in teams:
            for unit in self.state["teamStates"][team]["units"].values():
                state["teamStates"][team]["units"][unit.id] = {
                    "cargo": dict(unit.cargo),
                    "cooldown": unit.cooldown,
                    "x": unit.pos.x,
                    "y": unit.pos.y,
                    "type": unit.type,
                }

            state["teamStates"][team]["researchPoints"] = self.state["teamStates"][team]["researchPoints"]
            state["teamStates"][team]["researched"] = dict(self.state["teamStates"][team]["researched"])

        return state
