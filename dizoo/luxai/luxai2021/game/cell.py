"""
Implements /src/GameMap/cell.ts
"""

from .city import CityTile
from .position import Position
from .resource import Resource

"""
/**
 * Cell class for map cells
 *
 * Some restrictions not explicitly employed:
 * Cell can either be empty (no resource or citytile), or have a resource, or have a citytile, not multiple.
 * There may be multiple units but this is only allowed on city tiles
 */
"""


class Cell:
    def __init__(self, x, y, configs):
        """

        :param x:
        :param y:
        :param configs:
        """
        self.pos = Position(x, y)
        self.resource: Resource = None
        self.city_tile = None
        self.configs = configs
        self.units = {}
        self.road = configs["parameters"]["MIN_ROAD"]

    def set_resource(self, resource_type, amount):
        """
        
        :param resource_type: 
        :param amount: 
        :return: 
        """
        self.resource = Resource(resource_type, amount)

    def has_resource(self):
        """

        :return:
        """
        return self.resource is not None and self.resource.amount > 0

    def set_city_tile(self, team, city_id, cooldown=0.0):
        """
        
        :param team: 
        :param city_id: 
        :param cooldown: 
        :return: 
        """
        self.city_tile = CityTile(team, self.configs, cooldown)
        self.city_tile.pos = self.pos
        self.city_tile.city_id = city_id

    def is_city_tile(self):
        """

        :return:
        """
        return self.city_tile is not None

    def has_units(self):
        """

        :return:
        """
        return len(self.units) != 0

    def get_road(self):
        """

        :return:
        """
        if self.is_city_tile():
            return self.configs["parameters"]["MAX_ROAD"]
        else:
            return self.road
