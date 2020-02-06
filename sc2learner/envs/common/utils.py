from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pysc2.lib.unit_controls import Unit


def distance(a, b):

    def l2_dist(pos_a, pos_b):
        return ((pos_a[0] - pos_b[0]) ** 2 + (pos_a[1] - pos_b[1]) ** 2) ** 0.5

    if isinstance(a, Unit) and isinstance(b, Unit):
        return l2_dist((a.float_attr.pos_x, a.float_attr.pos_y),
                       (b.float_attr.pos_x, b.float_attr.pos_y))
    elif not isinstance(a, Unit) and isinstance(b, Unit):
        return l2_dist(a, (b.float_attr.pos_x, b.float_attr.pos_y))
    elif isinstance(a, Unit) and not isinstance(b, Unit):
        return l2_dist((a.float_attr.pos_x, a.float_attr.pos_y), b)
    else:
        return l2_dist(a, b)


def closest_unit(unit, target_units):
    assert len(target_units) > 0
    return min(target_units, key=lambda u: distance(unit, u))


def closest_units(unit, target_units, num):
    assert len(target_units) > 0
    return sorted(target_units, key=lambda u: distance(unit, u))[:num]


def closest_distance(unit, target_units):
    return min(distance(unit, u) for u in target_units) \
        if len(target_units) > 0 else float('inf')


def units_nearby(unit_center, target_units, max_distance):
    return [u for u in target_units if distance(unit_center, u) <= max_distance]


def strongest_health(units):
    return max(u.float_attr.health / u.float_attr.health_max for u in units)
