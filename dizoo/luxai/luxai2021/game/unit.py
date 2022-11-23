"""
Implements /src/Unit/index.ts -> Unit()
"""
import math

from .actionable import Actionable
from .actions import *
from .game_constants import GAME_CONSTANTS
from .position import Position

UNIT_TYPES = Constants.UNIT_TYPES


class Unit(Actionable):
    def __init__(self, x, y, unit_type, team, configs, idcount, cooldown=0.0, cargo=None):
        """

        :param x:
        :param y:
        :param unit_type:
        :param team:
        :param configs:
        :param idcount:
        :param cooldown:
        :param cargo:
        """
        super().__init__(configs, cooldown)
        if cargo is None:
            cargo = {"wood": 0, "uranium": 0, "coal": 0}
        self.pos = Position(x, y)
        self.team = team
        self.type = unit_type
        self.id = "u_%i" % idcount
        self.cargo = cargo
        self.can_act_override = None

    def is_worker(self) -> bool:
        return self.type == UNIT_TYPES.WORKER

    def is_cart(self) -> bool:
        return self.type == UNIT_TYPES.CART

    def get_cargo_space_left(self):
        """
        get cargo space left in this unit
        """
        space_used = self.cargo["wood"] + self.cargo["coal"] + self.cargo["uranium"]
        if self.type == UNIT_TYPES.WORKER:
            return GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["WORKER"] - space_used
        else:
            return GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"]["CART"] - space_used

    def get_cargo_fuel_value(self):
        """
        Returns the fuel-value of all the cargo this unit has.
        """
        return (
            self.cargo["wood"] * self.configs["parameters"]["RESOURCE_TO_FUEL_RATE"]["WOOD"] +
            self.cargo["coal"] * self.configs["parameters"]["RESOURCE_TO_FUEL_RATE"]["COAL"] +
            self.cargo["uranium"] * self.configs["parameters"]["RESOURCE_TO_FUEL_RATE"]["URANIUM"]
        )


    def spend_fuel_to_survive(self):
        """
        Implements /src/Unit/index.ts -> Unit.spendFuelToSurvive()
        """
        fuel_needed = self.get_light_upkeep()
        wood_needed = math.ceil(
            fuel_needed / self.configs["parameters"]["RESOURCE_TO_FUEL_RATE"]["WOOD"]
        )
        wood_used = min(self.cargo["wood"], wood_needed)
        fuel_needed -= wood_used * self.configs["parameters"]["RESOURCE_TO_FUEL_RATE"]["WOOD"]
        self.cargo["wood"] -= wood_used
        if fuel_needed <= 0:
            return True

        coal_needed = math.ceil(
            fuel_needed / self.configs["parameters"]["RESOURCE_TO_FUEL_RATE"]["COAL"]
        )
        coal_used = min(self.cargo["coal"], coal_needed)
        fuel_needed -= coal_used * self.configs["parameters"]["RESOURCE_TO_FUEL_RATE"]["COAL"]
        self.cargo["coal"] -= coal_used

        if fuel_needed <= 0:
            return True

        uranium_needed = math.ceil(
            fuel_needed / self.configs["parameters"]["RESOURCE_TO_FUEL_RATE"]["URANIUM"]
        )
        uranium_used = min(self.cargo["uranium"], uranium_needed)
        fuel_needed -= uranium_used * self.configs["parameters"]["RESOURCE_TO_FUEL_RATE"]["URANIUM"]
        self.cargo["uranium"] -= uranium_used

        if fuel_needed <= 0:
            return True

        return fuel_needed <= 0

    def can_build(self, game_map) -> bool:
        """
        whether or not the unit can build where it is right now
        """
        cell = game_map.get_cell_by_pos(self.pos)
        if not cell.has_resource() and self.can_act() and (
                self.cargo["wood"] + self.cargo["coal"] + self.cargo["uranium"]) >= GAME_CONSTANTS["PARAMETERS"][
            "CITY_BUILD_COST"]:
            return True
        return False


class Cargo:
    def __init__(self):
        self.wood = 0
        self.coal = 0
        self.uranium = 0

    def __str__(self) -> str:
        return f"Cargo | Wood: {self.wood}, Coal: {self.coal}, Uranium: {self.uranium}"


class Worker(Unit):
    """
    Worker class. Mirrors /src/Unit/index.ts -> Worker()
    """

    def __init__(self, x, y, team, configs, idcount, cooldown=0.0, cargo=None):
        if cargo is None:
            cargo = {"wood": 0, "uranium": 0, "coal": 0}
        super().__init__(x, y, Constants.UNIT_TYPES.WORKER, team, configs, idcount, cooldown, cargo)

    def get_light_upkeep(self):
        """

        :return:
        """
        return self.configs["parameters"]["LIGHT_UPKEEP"]["WORKER"]

    def can_move(self):
        """

        :return:
        """
        return self.can_act()

    def expend_resources_for_city(self):
        """

        :return:
        """
        # use wood, then coal, then uranium for building
        spent_resources = 0
        for rtype in ["wood", "coal", "uranium"]:
            if spent_resources + self.cargo[rtype] > self.configs["parameters"]["CITY_BUILD_COST"]:
                rtype_spent = self.configs["parameters"]["CITY_BUILD_COST"] - spent_resources
                self.cargo[rtype] -= rtype_spent
                break
            else:
                spent_resources += self.cargo[rtype]
                self.cargo[rtype] = 0

    def turn(self, game):
        """

        :param game:
        :return:
        """
        cell = game.map.get_cell_by_pos(self.pos)
        is_night = game.is_night()
        cooldown_multiplier = 2 if is_night else 1

        if len(self.current_actions) == 1:
            action = self.current_actions[0]
            acted = True
            if isinstance(action, MoveAction):
                game.move_unit(action.team, action.unit_id, action.direction)
            elif isinstance(action, TransferAction):
                game.transfer_resources(
                    action.team,
                    action.source_id,
                    action.destination_id,
                    action.resource_type,
                    action.amount
                )
            elif isinstance(action, SpawnCityAction):
                game.spawn_city_tile(action.team, self.pos.x, self.pos.y)
                self.expend_resources_for_city()
            elif isinstance(action, PillageAction):
                cell.road = max(
                    cell.road - self.configs["parameters"]["PILLAGE_RATE"],
                    self.configs["parameters"]["MIN_ROAD"]
                )
            else:
                acted = False

            if acted:
                self.cooldown += self.configs["parameters"]["UNIT_ACTION_COOLDOWN"]["WORKER"] * cooldown_multiplier


class Cart(Unit):
    """
    Cart class. Mirrors /src/Unit/index.ts -> Cart()
    """

    def __init__(self, x, y, team, configs, id_count, cooldown=0.0, cargo=None):
        """
        
        :param x: 
        :param y: 
        :param team: 
        :param configs: 
        :param id_count: 
        :param cooldown: 
        :param cargo: 
        """
        if cargo is None:
            cargo = {"wood": 0, "uranium": 0, "coal": 0}
        super().__init__(x, y, Constants.UNIT_TYPES.CART, team, configs, id_count, cooldown, cargo)

    def get_light_upkeep(self):
        """

        :return:
        """
        return self.configs["parameters"]["LIGHT_UPKEEP"]["CART"]

    def can_move(self):
        """

        :return:
        """
        return self.can_act()

    def turn(self, game):
        """

        :param game:
        :return:
        """
        cell = game.map.get_cell_by_pos(self.pos)
        is_night = game.is_night()
        cooldown_multiplier = 2 if is_night else 1

        if len(self.current_actions) == 1:
            action = self.current_actions[0]
            acted = True
            if isinstance(action, MoveAction):
                game.move_unit(action.team, action.unit_id, action.direction)
            elif isinstance(action, TransferAction):
                game.transfer_resources(
                    action.team,
                    action.source_id,
                    action.destination_id,
                    action.resource_type,
                    action.amount
                )
            self.cooldown += self.configs["parameters"]["UNIT_ACTION_COOLDOWN"]["CART"] * cooldown_multiplier

        end_cell = game.map.get_cell_by_pos(self.pos)

        # auto create roads by increasing the cooldown value of the the cell unit is on currently
        if end_cell.get_road() < self.configs["parameters"]["MAX_ROAD"]:
            end_cell.road = min(
                end_cell.road + self.configs["parameters"]["CART_ROAD_DEVELOPMENT_RATE"],
                self.configs["parameters"]["MAX_ROAD"]
            )
            game.stats["teamStats"][self.team]["roadsBuilt"] += self.configs["parameters"]["CART_ROAD_DEVELOPMENT_RATE"]
            if end_cell not in game.cells_with_roads:
                game.cells_with_roads.add(end_cell)
