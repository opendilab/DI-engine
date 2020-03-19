from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import math

import numpy as np
from pysc2.lib.typeenums import UNIT_TYPEID as UNIT_TYPE

import sc2learner.envs.common.utils as utils
from sc2learner.envs.common.const import PLACE_COLLISION_BUILDINGS


class Placer(object):
    def get_building_position(self, type_id, dc):
        if type_id == UNIT_TYPE.ZERG_HATCHERY.value:
            return self._next_base_place(dc)
        elif type_id == UNIT_TYPE.ZERG_EXTRACTOR.value:
            gas = dc.exploitable_gas
            return random.choice(gas) if len(gas) > 0 else None
        else:
            place = self._constructable_place(1.5, dc)
            return random.choice(place) if len(place) > 0 else None

    def can_build(self, type_id, dc):
        if type_id == UNIT_TYPE.ZERG_HATCHERY.value:
            return self._next_base_place(dc) is not None
        elif type_id == UNIT_TYPE.ZERG_EXTRACTOR.value:
            return len(dc.exploitable_gas) > 0
        else:
            place = self._constructable_place(1.5, dc)
            return len(place) > 0

    def _constructable_place(self, margin, dc):
        place = []
        bases = dc.mature_units_of_types(
            [UNIT_TYPE.ZERG_HATCHERY.value, UNIT_TYPE.ZERG_LAIR.value, UNIT_TYPE.ZERG_HIVE.value]
        )
        for base in bases:
            search_region = (base.float_attr.pos_x - 10.5, base.float_attr.pos_y - 10.5, 10.5 * 2, 10.5 * 2)
            place.extend(self._search_place(search_region, dc, margin=margin, remove_corner=True, expand_mineral=True))
        return place

    def _next_base_place(self, dc):
        unexploited_minerals = dc.unexploited_minerals
        if len(unexploited_minerals) == 0:
            return None
        mineral_to_exploit = utils.closest_unit(dc.init_base_pos, unexploited_minerals)
        resources_nearby = utils.units_nearby(mineral_to_exploit, dc.minerals + dc.gas, max_distance=14)
        x_list = [u.float_attr.pos_x for u in resources_nearby]
        y_list = [u.float_attr.pos_y for u in resources_nearby]
        x_mean = sum(x_list) / len(x_list)
        y_mean = sum(y_list) / len(y_list)
        left = int(math.floor(min(x_list)))
        right = int(math.ceil(max(x_list)))
        bottom = int(math.floor(min(y_list)))
        top = int(math.ceil(max(y_list)))
        width = right - left + 1
        height = top - bottom + 1
        x_offset, y_offset = 0, 0
        if height - width >= 5:
            left_mid = (left, (bottom + top) / 2)
            right_mid = (right, (bottom + top) / 2)
            if utils.closest_distance(left_mid, resources_nearby) > \
                    utils.closest_distance(right_mid, resources_nearby):
                x_offset = width - height + 1
            width = height - 1
        elif height - width <= -5:
            top_mid = ((left + right) / 2, top)
            bottom_mid = ((left + right) / 2, bottom)
            if utils.closest_distance(top_mid, resources_nearby) < \
                    utils.closest_distance(bottom_mid, resources_nearby):
                y_offset = height - width + 1
            height = width - 1
        region = [left + x_offset, bottom + y_offset, width, height]
        place = self._search_place(region, dc, margin=5.5, shrink_mineral=True)
        return utils.closest_unit((x_mean, y_mean), place) \
            if len(place) > 0 else None

    def _search_place(
        self, search_region, dc, margin=0, remove_corner=False, expand_mineral=False, shrink_mineral=False
    ):
        bottomleft = tuple(map(int, search_region[:2]))
        size = tuple(map(int, search_region[2:]))
        grids = np.zeros(size).astype(np.int)
        if remove_corner:
            cx, cy = size[0] / 2.0, size[1] / 2.0
            r = max(size[0] / 2.0, size[1] / 2.0)
            r_sqrt = (r - 0.5)**2
            for x in range(size[0]):
                x_sqrt = (x + 0.5 - cx)**2
                for y in range(size[1]):
                    y_sqrt = (y + 0.5 - cy)**2
                    if x_sqrt + y_sqrt > r_sqrt:
                        grids[x, y] = 1
        filter_range = (
            bottomleft[0] - 10, bottomleft[0] + size[0] + 10, bottomleft[1] - 10, bottomleft[1] + size[1] + 10
        )
        sitted_units = [
            u for u in dc.units if (
                u.unit_type in PLACE_COLLISION_BUILDINGS and u.float_attr.pos_x >= filter_range[0]
                and u.float_attr.pos_x <= filter_range[1] and u.float_attr.pos_y >= filter_range[2]
                and u.float_attr.pos_y <= filter_range[3]
            )
        ]
        sitted_units = [u for u in dc.units if u.unit_type in PLACE_COLLISION_BUILDINGS]
        margin_int = math.floor(margin)
        for u in sitted_units:
            if u.float_attr.radius <= 1.0:
                r_x = r_y = 1.0
                if u.float_attr.pos_x % 1 != 0:
                    r_x -= 0.5
                if u.float_attr.pos_y % 1 != 0:
                    r_y -= 0.5
            else:
                r_x = r_y = float(int(u.float_attr.radius))
                if u.float_attr.pos_x % 1 != 0:
                    r_x += 0.5
                if u.float_attr.pos_y % 1 != 0:
                    r_y += 0.5
            if (shrink_mineral
                    and u.unit_type in {UNIT_TYPE.NEUTRAL_MINERALFIELD.value, UNIT_TYPE.NEUTRAL_MINERALFIELD750.value}):
                if r_x == 1.5 and r_y == 1:
                    r_x = 0.5
                elif r_x == 1 and r_y == 1.5:
                    r_y = 0.5
            if (expand_mineral
                    and u.unit_type in {UNIT_TYPE.NEUTRAL_MINERALFIELD.value, UNIT_TYPE.NEUTRAL_MINERALFIELD750.value}):
                r_x += 1
                r_y += 1
            r_x += margin_int
            r_y += margin_int
            xl = int(u.float_attr.pos_x - r_x - bottomleft[0])
            xr = int(u.float_attr.pos_x + r_x - bottomleft[0])
            yu = int(u.float_attr.pos_y - r_y - bottomleft[1])
            yd = int(u.float_attr.pos_y + r_y - bottomleft[1])
            grids[max(xl, 0):max(min(xr, size[0]), 0), max(yu, 0):max(min(yd, size[1]), 0)] = 1

        slopes = [(76.5, 90.5), (73.5, 86.5), (123.5, 52.5), (126.5, 56.5), (131.5, 36.5), (68.5, 106.5)]
        holes = [(124.5, 34.5), (76.5, 108.5), (154.5, 60.5), (45.5, 82.5)]
        r = 2.5
        for x, y in slopes + holes:
            xl = int(x - r - bottomleft[0])
            xr = int(x + r - bottomleft[0])
            yu = int(y - r - bottomleft[1])
            yd = int(y + r - bottomleft[1])
            grids[max(xl, 0):max(min(xr, size[0]), 0), max(yu, 0):max(min(yd, size[1]), 0)] = 1
        x, y = np.nonzero(1 - grids)
        # if remove_corner == True:
        #np.set_printoptions(threshold=np.nan, linewidth=300)
        # print(grids)
        return list(zip(x + bottomleft[0] + 0.5, y + bottomleft[1] + 0.5))
