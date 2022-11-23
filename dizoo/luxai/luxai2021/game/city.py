"""
Implements /src/Game/city.ts
"""

from .actionable import Actionable
from .actions import *

"""
//**
 * A city is composed of adjacent city tiles of the same team
 */
 """


class City:
    def __init__(self, team, configs, id_count, city_id=None, fuel=0):
        """

        :param team:
        :param configs:
        :param id_count:
        :param city_id:
        :param fuel:
        """
        self.team = team
        self.configs = configs
        if city_id:
            self.id = city_id
        else:
            self.id = "c_%i" % id_count
        self.fuel = fuel
        self.city_cells = []

    def get_light_upkeep(self):
        """

        :return:
        """
        return len(self.city_cells) * self.configs["parameters"]["LIGHT_UPKEEP"]["CITY"] - self.get_adjacency_bonuses()

    def get_adjacency_bonuses(self):
        """

        :return:
        """
        bonus = 0
        for cell in self.city_cells:
            bonus += cell.city_tile.adjacent_city_tiles * self.configs["parameters"]["CITY_ADJACENCY_BONUS"]

        return bonus

    def add_city_tile(self, cell):
        """

        :param cell:
        """
        self.city_cells.append(cell)


class CityTile(Actionable):
    def __init__(self, team, configs, cooldown=0.0) -> None:
        self.team = team
        self.pos = None
        self.city_id = None
        self.adjacent_city_tiles = 0
        super().__init__(configs, cooldown)

    def get_tile_id(self):
        """

        :return:
        """
        return f"{self.city_id}_{self.pos.x}_{self.pos.y}"

    def can_build_unit(self):
        """

        :return:
        """
        return self.can_act()

    def can_research(self):
        """

        :return:
        """
        return self.can_act()

    def get_cargo_space_left(self):
        """

        :return:
        """
        return 9999999  # Infinite space

    def turn(self, game):
        """

        :param game:
        """
        if len(self.current_actions) == 1:
            action = self.current_actions[0]
            if isinstance(action, SpawnCartAction):
                game.spawn_cart(action.team, action.x, action.y)
                self.reset_cooldown()
            elif isinstance(action, SpawnWorkerAction):
                game.spawn_worker(action.team, action.x, action.y)
                self.reset_cooldown()
            elif isinstance(action, ResearchAction):
                self.reset_cooldown()
                game.state["teamStates"][self.team]["researchPoints"] += 1
                if (game.state["teamStates"][self.team]["researchPoints"] >=
                        self.configs["parameters"]["RESEARCH_REQUIREMENTS"]["COAL"]):
                    game.state["teamStates"][self.team]["researched"]["coal"] = True
                if (game.state["teamStates"][self.team]["researchPoints"] >=
                        self.configs["parameters"]["RESEARCH_REQUIREMENTS"]["URANIUM"]):
                    game.state["teamStates"][self.team]["researched"]["uranium"] = True

        if self.cooldown > 0:
            self.cooldown -= 1

    def reset_cooldown(self):
        """

        """
        self.cooldown = self.configs["parameters"]["CITY_ACTION_COOLDOWN"]
