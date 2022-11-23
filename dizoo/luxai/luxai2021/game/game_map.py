import math
import random
from typing import List

from argparse import Namespace
from ..env.rng.rng import get_n_values
from .cell import Cell
from .constants import Constants
from .position import Position

DIRECTIONS = Constants.DIRECTIONS
RESOURCE_TYPES = Constants.RESOURCE_TYPES

""" Enum implemenations """
mapSizes = [12, 16, 24, 32]


class SYMMETRY:
    HORIZONTAL = 0
    VERTICAL = 1


MOVE_DELTAS = [
    [0, 1],
    [-1, 1],
    [-1, 0],
    [-1, -1],
    [0, -1],
    [1, -1],
    [1, 0],
    [1, 1],
]


def sign(value):
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


"""Implements /src/GameMap/index.ts"""


class GameMap:
    def __init__(self, configs):
        """

        :param configs:
        """
        self.configs = configs
        self.resources = []
        self.resources_by_type = {
            Constants.RESOURCE_TYPES.WOOD: [],
            Constants.RESOURCE_TYPES.COAL: [],
            Constants.RESOURCE_TYPES.URANIUM: [],
        }

    def generate_map(self, game):
        """
        Initialize the random map
        Implements /src/Game/gen.ts
        :param game:
        """
        
        def js_rng(seed):
            idx = 0
            rng_values = get_n_values(seed, N=1000000)
            def _rng():
                nonlocal idx
                ret = rng_values[idx]
                idx += 1
                return ret
            return Namespace(**dict(random=_rng))
        

        if self.configs["seed"] is not None:
            # Use a random number generator that exactly matches LuxAI. That way
            # the same seeds generate the exact same map.
            seed = self.configs["seed"]
            rng = js_rng(seed)
        else:
            rng = random.Random()

        size = mapSizes[math.floor(rng.random() * len(mapSizes))]

        if "width" in self.configs:
            self.width = self.configs["width"]
        else:
            self.width = size
        
        if "height" in self.configs:
            self.height = self.configs["height"]
        else:
            self.height = size

        # Create map tiles
        self.map: List[List[Cell]] = [None] * self.height
        for y in range(0, self.height):
            self.map[y] = [None] * self.width
            for x in range(0, self.width):
                self.map[y][x] = Cell(x, y, self.configs)

        if self.configs["mapType"] == Constants.MAP_TYPES.EMPTY:
            return
        else:
            symmetry = SYMMETRY.HORIZONTAL
            half_width = self.width
            half_height = self.height
            if rng.random() < 0.5:
                symmetry = SYMMETRY.VERTICAL
                half_width = math.floor(self.width / 2)
            else:
                half_height = math.floor(self.height / 2)

            resources_map = self._generate_all_resources(
                rng,
                symmetry,
                self.width,
                self.height,
                half_width,
                half_height
            )

            retries = 0
            while not self._validate_resources_map(resources_map):
                retries += 1
                resources_map = self._generate_all_resources(
                    rng,
                    symmetry,
                    self.width,
                    self.height,
                    half_width,
                    half_height
                )

            for y, row in enumerate(resources_map):
                for x, val in enumerate(row):
                    if val is not None:
                        self.add_resource(x, y, val["type"], val["amt"])

            spawn_x = math.floor(rng.random() * (half_width - 1)) + 1
            spawn_y = math.floor(rng.random() * (half_height - 1)) + 1
            while self.get_cell(spawn_x, spawn_y).has_resource():
                spawn_x = math.floor(rng.random() * (half_width - 1)) + 1
                spawn_y = math.floor(rng.random() * (half_height - 1)) + 1

            game.spawn_worker(Constants.TEAM.A, spawn_x, spawn_y)
            game.spawn_city_tile(Constants.TEAM.A, spawn_x, spawn_y)
            if symmetry == SYMMETRY.HORIZONTAL:
                game.spawn_worker(Constants.TEAM.B, spawn_x, self.height - spawn_y - 1)
                game.spawn_city_tile(Constants.TEAM.B, spawn_x, self.height - spawn_y - 1)
            else:
                game.spawn_worker(Constants.TEAM.B, self.width - spawn_x - 1, spawn_y)
                game.spawn_city_tile(Constants.TEAM.B, self.width - spawn_x - 1, spawn_y)

            # add at least 3 wood deposits near spawns
            delta_index = math.floor(rng.random() * len(MOVE_DELTAS))
            wood_spawns_deltas = [
                MOVE_DELTAS[delta_index],
                MOVE_DELTAS[(delta_index + 1) % len(MOVE_DELTAS)],
                MOVE_DELTAS[(delta_index + 2) % len(MOVE_DELTAS)],
                MOVE_DELTAS[(delta_index + 3) % len(MOVE_DELTAS)],
                MOVE_DELTAS[(delta_index + 4) % len(MOVE_DELTAS)],
                MOVE_DELTAS[(delta_index + 5) % len(MOVE_DELTAS)],
                MOVE_DELTAS[(delta_index + 6) % len(MOVE_DELTAS)]
            ]
            count = 0
            for delta in wood_spawns_deltas:
                nx = spawn_x + delta[0]
                ny = spawn_y + delta[1]
                nx2 = nx
                ny2 = ny
                if symmetry == SYMMETRY.HORIZONTAL:
                    ny2 = self.height - ny - 1
                else:
                    nx2 = self.width - nx - 1

                if not self.in_map(Position(nx, ny)) or not self.in_map(Position(nx2, ny2)):
                    continue

                if not self.get_cell(nx, ny).has_resource() and self.get_cell(nx, ny).city_tile is None:
                    count += 1
                    self.add_resource(nx, ny, Constants.RESOURCE_TYPES.WOOD, 800)

                if not self.get_cell(nx2, ny2).has_resource() and self.get_cell(nx2, ny2).city_tile is None:
                    count += 1
                    self.add_resource(nx2, ny2, Constants.RESOURCE_TYPES.WOOD, 800)

                if count == 6:
                    break

            return

    def _validate_resources_map(self, resources_map):
        """

        :param resources_map:
        :return:
        """
        data = {"wood": 0, "coal": 0, "uranium": 0}
        for y, row in enumerate(resources_map):
            for x, val in enumerate(row):
                if val is not None:
                    data[resources_map[y][x]["type"]] += resources_map[y][x]["amt"]

        if data["wood"] < 2000:
            return False
        if data["coal"] < 1500:
            return False
        if data["uranium"] < 300:
            return False
        return True

    def _generate_all_resources(self, rng, symmetry, width, height, half_width, half_height):
        """

        :param rng:
        :param symmetry:
        :param width:
        :param height:
        :param half_width:
        :param half_height:
        :return:
        """
        resources_map = []

        for i in range(height):
            resources_map.append([])
            for j in range(width):
                resources_map[i].append(None)

        wood_resources_map = self._generate_resource_map(
            rng,
            0.21,
            0.01,
            half_width,
            half_height,
            {"deathLimit": 2, "birthLimit": 4}
        )

        for y, row in enumerate(wood_resources_map):
            for x, val in enumerate(row):
                if val == 1:
                    amt = min(300 + math.floor(rng.random() * 100), 500)
                    resources_map[y][x] = {"type": Constants.RESOURCE_TYPES.WOOD, "amt": amt}

        coal_resources_map = self._generate_resource_map(
            rng,
            0.11,
            0.02,
            half_width,
            half_height,
            {"deathLimit": 2, "birthLimit": 4}
        )

        for y, row in enumerate(coal_resources_map):
            for x, val in enumerate(row):
                if val == 1:
                    amt = 350 + math.floor(rng.random() * 75)
                    resources_map[y][x] = {"type": Constants.RESOURCE_TYPES.COAL, "amt": amt}

        uranium_resources_map = self._generate_resource_map(
            rng,
            0.055,
            0.04,
            half_width,
            half_height,
            {"deathLimit": 1, "birthLimit": 6}
        )

        for y, row in enumerate(uranium_resources_map):
            for x, val in enumerate(row):
                if val == 1:
                    amt = 300 + math.floor(rng.random() * 50)
                    resources_map[y][x] = {"type": Constants.RESOURCE_TYPES.URANIUM, "amt": amt}

        for i in range(10):
            resources_map = self._gravitate_resources(resources_map)

        # perturb resources
        for y in range(half_height):
            for x in range(half_width):
                resource = resources_map[y][x]
                if resource is None:
                    continue
                for d in MOVE_DELTAS:
                    nx = x + d[0]
                    ny = y + d[1]
                    if nx < 0 or ny < 0 or nx >= half_height or ny >= half_width:
                        continue
                    if rng.random() < 0.05:
                        amt = 300 + math.floor(rng.random() * 50)
                        if resource["type"] == 'coal':
                            amt = 350 + math.floor(rng.random() * 75)

                        if resource["type"] == 'wood':
                            amt = min(300 + math.floor(rng.random() * 100), 500)

                        resources_map[ny][nx] = {"type": resource["type"], "amt": amt}

        for y in range(half_height):
            for x in range(half_width):
                resource = resources_map[y][x]
                if symmetry == SYMMETRY.VERTICAL:
                    resources_map[y][width - x - 1] = resource
                else:
                    resources_map[height - y - 1][x] = resource

        return resources_map

    def _generate_resource_map(self, rng, density, density_range, width, height,
                               gol_options={"deathLimit": 2, "birthLimit": 4}):
        # width, height should represent half of the map
        local_density = density - density_range / 2 + density_range * rng.random()
        arr = []
        for y in range(height):
            arr.append([])
            for x in range(width):
                resource_type = 0
                if rng.random() < local_density:
                    resource_type = 1

                arr[y].append(resource_type)

        # simulate GOL for 2 rounds
        for i in range(2):
            arr = self._simulate_gol(arr, gol_options)

        return arr

    def _simulate_gol(self, arr, options):
        """

        :param arr:
        :param options:
        :return:
        """
        # high birthlimit = unlikely to deviate from initial random spots
        # high deathlimit = lots of patches die
        padding = 1
        death_limit = options["deathLimit"]
        birth_limit = options["birthLimit"]
        for i in range(padding, len(arr) - padding):
            for j in range(padding, len(arr[0]) - padding):
                alive = 0
                for k in range(len(MOVE_DELTAS)):
                    delta = MOVE_DELTAS[k]
                    ny = i + delta[1]
                    nx = j + delta[0]
                    if arr[ny][nx] == 1:
                        alive += 1

                if arr[i][j] == 1:
                    if alive < death_limit:
                        arr[i][j] = 0
                    else:
                        arr[i][j] = 1
                else:
                    if alive > birth_limit:
                        arr[i][j] = 1
                    else:
                        arr[i][j] = 0

        return arr

    def _kernel_force(self, resources_map, rx, ry):
        """

        :param resources_map:
        :param rx:
        :param ry:
        :return:
        """
        force = [0, 0]
        resource = resources_map[ry][rx]
        kernel_size = 5

        for y in range(ry - kernel_size, ry + kernel_size):
            for x in range(rx - kernel_size, rx + kernel_size):
                if x < 0 or y < 0 or x >= len(resources_map[0]) or y >= len(resources_map): continue

                r2 = resources_map[y][x]
                if r2 is not None:
                    dx = rx - x
                    dy = ry - y
                    mdist = abs(dx) + abs(dy)
                    if r2["type"] != resource["type"]:
                        if dx != 0:
                            force[0] += math.pow(dx / mdist, 2) * sign(dx)
                        if dy != 0:
                            force[1] += math.pow(dy / mdist, 2) * sign(dy)
                    else:
                        if dx != 0:
                            force[0] -= math.pow(dx / mdist, 2) * sign(dx)
                        if dy != 0:
                            force[1] -= math.pow(dy / mdist, 2) * sign(dy)

        return force

    def _gravitate_resources(self, resources_map):
        """

        :param resources_map:
        :return:
        """
        #
        # Gravitate like to like, push different resources away from each other.
        # 
        # Add's a force direction to each cell.
        #
        new_resources_map = []
        for y in range(len(resources_map)):
            new_resources_map.append([])
            for x in range(len(resources_map[y])):
                new_resources_map[y].append(None)
                res = resources_map[y][x]
                if res is not None:
                    f = self._kernel_force(resources_map, x, y)
                    resources_map[y][x]["force"] = f

        for y in range(len(resources_map)):
            for x in range(len(resources_map[y])):
                res = resources_map[y][x]
                if res is not None:
                    nx = x + sign(res["force"][0]) * 1
                    ny = y + sign(res["force"][1]) * 1
                    if nx < 0: nx = 0
                    if ny < 0: ny = 0
                    if nx >= len(resources_map[0]): nx = len(resources_map[0]) - 1
                    if ny >= len(resources_map): ny = len(resources_map) - 1
                    if new_resources_map[ny][nx] is None:
                        new_resources_map[ny][nx] = res
                    else:
                        new_resources_map[y][x] = res

        return new_resources_map

    def add_resource(self, x, y, resource_type, amount):
        """

        :param x:
        :param y:
        :param resource_type:
        :param amount:
        :return:
        """
        cell = self.get_cell(x, y)
        cell.set_resource(resource_type, amount)
        self.resources.append(cell)
        self.resources_by_type[resource_type].append(cell)
        return cell

    def get_cell_by_pos(self, pos) -> Cell:
        """

        :param pos:
        :return:
        """
        if pos.y >= len(self.map) or pos.x >= len(self.map[0]) or pos.y < 0 or pos.x < 0:
            return None
        return self.map[pos.y][pos.x]

    def get_cell(self, x, y) -> Cell:
        """

        :param x:
        :param y:
        :return:
        """
        if y >= len(self.map) or x >= len(self.map[0]) or y < 0 or x < 0:
            return None
        return self.map[y][x]

    def get_row(self, y):
        """

        :param y:
        :return:
        """
        return self.map[y]

    def get_adjacent_cells(self, cell):
        """

        :param cell:
        :return:
        """
        cells = []

        # NORTH
        if cell.pos.y > 0:
            cells.append(self.get_cell(cell.pos.x, cell.pos.y - 1))

        # EAST
        if cell.pos.x < self.width - 1:
            cells.append(self.get_cell(cell.pos.x + 1, cell.pos.y))

        # SOUTH
        if cell.pos.y < self.height - 1:
            cells.append(self.get_cell(cell.pos.x, cell.pos.y + 1))

        # WEST
        if cell.pos.x > 0:
            cells.append(self.get_cell(cell.pos.x - 1, cell.pos.y))

        return cells

    def get_adjacent_cells_with_corners(self, cell):
        """
        Includes the corners as 'adjacent'. Used in finding
        resource clusters.
        :param cell:
        :return:
        """
        cells = self.get_adjacent_cells(cell)

        c = self.get_cell(cell.pos.x - 1, cell.pos.y - 1)
        if c:
            cells.append(c)

        c = self.get_cell(cell.pos.x + 1, cell.pos.y - 1)
        if c:
            cells.append(c)

        c = self.get_cell(cell.pos.x - 1, cell.pos.y + 1)
        if c:
            cells.append(c)

        c = self.get_cell(cell.pos.x + 1, cell.pos.y + 1)
        if c:
            cells.append(c)

        return cells

    def in_map(self, pos):
        """

        :param pos:
        :return:
        """
        return not (pos.x < 0 or pos.y < 0 or pos.x >= self.width or pos.y >= self.height)

    """
    * Return printable map string
    """

    def get_map_string(self):
        """

        :return:
        """
        # W<team> = Worker
        # C<team> = Cart
        # ◰<team> = City
        # <number><team> = Stacked units from specified team.
        # ▩▩ = Wood
        # ▣▣ = Coal
        # ▷▷ == Uranium
        map_str = ''
        for y in range(self.height):
            row = self.get_row(y)
            for cell in row:
                if cell.has_units():
                    unit = list(cell.units.values())[0]
                    if len(cell.units) == 1:
                        unit_str = '?'
                        if unit.type == Constants.UNIT_TYPES.CART:
                            unit_str = 'c'
                        elif unit.type == Constants.UNIT_TYPES.WORKER:
                            unit_str = 'W'

                        if unit.team == Constants.TEAM.A:
                            unit_str += "a"
                        elif unit.team == Constants.TEAM.B:
                            unit_str += "b"
                        else:
                            unit_str += "?"

                        map_str += unit_str
                    else:
                        unit_str = str(len(cell.units))

                        if unit.team == Constants.TEAM.A:
                            unit_str += "a"
                        elif unit.team == Constants.TEAM.B:
                            unit_str += "b"
                        else:
                            unit_str += "?"

                        map_str += unit_str
                elif cell.has_resource():
                    if cell.resource.type == Constants.RESOURCE_TYPES.WOOD:
                        map_str += "w,"
                    if cell.resource.type == Constants.RESOURCE_TYPES.COAL:
                        map_str += "c,"
                    if cell.resource.type == Constants.RESOURCE_TYPES.URANIUM:
                        map_str += "u,"
                elif cell.is_city_tile():
                    map_str += "C"
                    if cell.city_tile.team == Constants.TEAM.A:
                        map_str += "a"
                    elif cell.city_tile.team == Constants.TEAM.B:
                        map_str += "b"
                    else:
                        map_str += "?"
                else:
                    map_str += ".."
            map_str += "\n"
        return map_str

    def to_state_object(self):
        """
        Implements /src/GameMap/index.ts -> toStateObject()
        """
        obj = []
        for y in range(self.height):
            obj.append([])
            for x in range(self.width):
                cell = self.get_cell(x, y);
                cell_data = {}
                
                if cell.get_road() != 0:
                    cell_data["road"] = cell.get_road()
                
                if cell.resource:
                    cell_data["type"] = cell.resource.type
                    cell_data["amount"] = cell.resource.amount
                
                obj[y].append(cell_data)
        
        return obj
