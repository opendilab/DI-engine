import enum
import math

import numpy as np
from collections import namedtuple
from s2clientprotocol import common_pb2 as sc_common, sc2api_pb2 as sc_pb, raw_pb2 as r_pb

ORIGINAL_AGENT = "me"
OPPONENT_AGENT = "opponent"

MOVE_EAST = 4
MOVE_WEST = 5

actions = {
    "move": 16,  # target: PointOrUnit
    "attack": 23,  # target: PointOrUnit
    "stop": 4,  # target: None
    "heal": 386,  # Unit
    "parasitic_bomb": 2542,  # target: Unit
    'fungal_growth': 74,  # target: PointOrUnit
}


class Direction(enum.IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3


def distance(x1, y1, x2, y2):
    """Distance between two points."""
    return math.hypot(x2 - x1, y2 - y1)


class SMACAction:
    info_template = namedtuple('EnvElementInfo', ['shape', 'value', 'to_agent_processor', 'from_agent_processor'])

    def __init__(self, n_agents, n_enemies, two_player=False, mirror_opponent=True):
        self.obs_pathing_grid = False
        self.obs_terrain_height = False
        self.state_last_action = True
        self.state_timestep_number = False
        self.n_obs_pathing = 8
        self.n_obs_height = 9
        self._move_amount = 2
        self.n_actions_no_attack = 6
        self.n_actions_move = 4
        self.n_actions = self.n_actions_no_attack + n_enemies
        self.map_x = 0
        self.map_y = 0

        # Status tracker
        self.last_action = np.zeros((n_agents, self.n_actions))
        self.last_action_opponent = np.zeros((n_enemies, self.n_actions))
        self.n_agents = n_agents
        self.n_enemies = n_enemies

        self.two_player = two_player
        self.mirror_opponent = mirror_opponent

    def reset(self):
        self.last_action.fill(0)
        self.last_action_opponent.fill(0)

    def update(self, map_info, map_x, map_y):
        if map_info.pathing_grid.bits_per_pixel == 1:
            vals = np.array(list(map_info.pathing_grid.data)).reshape(map_x, int(map_y / 8))
            self.pathing_grid = np.transpose(
                np.array([[(b >> i) & 1 for b in row for i in range(7, -1, -1)] for row in vals], dtype=np.bool)
            )
        else:
            self.pathing_grid = np.invert(
                np.flip(
                    np.transpose(np.array(list(map_info.pathing_grid.data), dtype=np.bool).reshape(map_x, map_y)),
                    axis=1
                )
            )

        self.terrain_height = np.flip(
            np.transpose(np.array(list(map_info.terrain_height.data)).reshape(map_x, map_y)), 1
        ) / 255
        self.map_x = map_x
        self.map_y = map_y

    def _parse_single(self, actions, engine, is_opponent=False):
        actions = np.asarray(actions, dtype=np.int)
        assert len(actions) == (self.n_enemies if is_opponent else self.n_agents)

        actions_int = [int(a) for a in actions]
        # Make them one-hot
        if is_opponent:
            self.last_action_opponent = np.eye(self.n_actions)[np.array(actions_int)]
        else:
            self.last_action = np.eye(self.n_actions)[np.array(actions_int)]

        sc_actions = []
        for a_id, action in enumerate(actions_int):
            sc_action = self.get_agent_action(a_id, action, engine, is_opponent)
            if sc_action:
                sc_actions.append(sc_action)
        return sc_actions

    def get_action(self, actions, engine):
        if self.two_player:
            # ========= Two player mode ==========
            assert self.two_player
            assert isinstance(actions, dict)
            assert ORIGINAL_AGENT in actions
            assert OPPONENT_AGENT in actions

            if self.mirror_opponent:
                actions[OPPONENT_AGENT] = [self._transform_action(a) for a in actions[OPPONENT_AGENT]]

            sc_actions_me = self._parse_single(actions[ORIGINAL_AGENT], engine, is_opponent=False)
            sc_actions_opponent = self._parse_single(actions[OPPONENT_AGENT], engine, is_opponent=True)

            return {ORIGINAL_AGENT: sc_actions_me, OPPONENT_AGENT: sc_actions_opponent}
        else:
            assert not isinstance(actions, dict)
            sc_actions = self._parse_single(actions, engine, is_opponent=False)
            return sc_actions

    def get_unit_by_id(self, a_id, engine, is_opponent=False):
        """Get unit by ID."""
        if is_opponent:
            return engine.enemies[a_id]
        return engine.agents[a_id]

    def get_agent_action(self, a_id, action, engine, is_opponent=False):
        """Construct the action for agent a_id.
        The input action here is *absolute* and is not mirrored!
        We use skip_mirror=True in get_avail_agent_actions to avoid error.
        """
        avail_actions = self.get_avail_agent_actions(a_id, engine, is_opponent=is_opponent, skip_mirror=True)
        try:
            assert avail_actions[action] == 1, \
                "Agent {} cannot perform action {} in ava {}".format(a_id, action, avail_actions)
        except Exception as e:
            if action == 0:
                action = 1
            else:
                action = 1
                # TODO
                # raise e
        unit = self.get_unit_by_id(a_id, engine, is_opponent=is_opponent)

        # if is_opponent:
        #     action = avail_actions[0] if avail_actions[0] else avail_actions[1]

        # ===== The follows is intact to the original =====
        tag = unit.tag
        type_id = unit.unit_type
        x = unit.pos.x
        y = unit.pos.y

        # if is_opponent:
        #     print(f"The given unit tag {tag}, x {x}, y {y} and action {action}")

        if action == 0:
            # no-op (valid only when dead)
            assert unit.health == 0, "No-op only available for dead agents."
            return None
        elif action == 1:
            # stop
            cmd = r_pb.ActionRawUnitCommand(ability_id=actions["stop"], unit_tags=[tag], queue_command=False)

        elif action == 2:
            # move north
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(x=x, y=y + self._move_amount),
                unit_tags=[tag],
                queue_command=False
            )

        elif action == 3:
            # move south
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(x=x, y=y - self._move_amount),
                unit_tags=[tag],
                queue_command=False
            )

        elif action == 4:
            # move east
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(x=x + self._move_amount, y=y),
                unit_tags=[tag],
                queue_command=False
            )

        elif action == 5:
            # move west
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(x=x - self._move_amount, y=y),
                unit_tags=[tag],
                queue_command=False
            )
        else:
            # attack/heal units that are in range
            target_id = action - self.n_actions_no_attack
            if engine.map_type == "MMM" and unit.unit_type == (engine.medivac_id_opponent
                                                               if is_opponent else engine.medivac_id):
                target_unit = (engine.enemies[target_id] if is_opponent else engine.agents[target_id])
                action_name = "heal"
            elif engine.map_type == 'infestor_viper':
                # viper
                if type_id == 499:
                    target_unit = engine.enemies[target_id]
                    action_name = "parasitic_bomb"
                # infestor
                else:
                    target_unit = engine.enemies[target_id]
                    target_loc = (target_unit.pos.x, target_unit.pos.y)
                    action_name = "fungal_growth"
                    target_loc = sc_common.Point2D(x=target_loc[0], y=target_loc[1])
                    cmd = r_pb.ActionRawUnitCommand(
                        ability_id=actions[action_name],
                        target_world_space_pos=target_loc,
                        unit_tags=[tag],
                        queue_command=False
                    )
                    return sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=cmd))
            else:
                target_unit = (engine.agents[target_id] if is_opponent else engine.enemies[target_id])
                action_name = "attack"

            action_id = actions[action_name]
            target_tag = target_unit.tag

            cmd = r_pb.ActionRawUnitCommand(
                ability_id=action_id, target_unit_tag=target_tag, unit_tags=[tag], queue_command=False
            )

        sc_action = sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=cmd))
        return sc_action

    def get_avail_agent_actions(self, agent_id, engine, is_opponent=False, skip_mirror=False):
        """Returns the available actions for agent_id."""
        medivac_id = engine.medivac_id_opponent if is_opponent else engine.medivac_id
        unit = self.get_unit_by_id(agent_id, engine, is_opponent)
        if unit.health > 0:
            # cannot choose no-op when alive
            avail_actions = [0] * self.n_actions

            # stop should be allowed
            avail_actions[1] = 1

            # see if we can move
            if self.can_move(unit, Direction.NORTH):
                avail_actions[2] = 1
            if self.can_move(unit, Direction.SOUTH):
                avail_actions[3] = 1
            if self.can_move(unit, Direction.EAST):
                avail_actions[4] = 1
            if self.can_move(unit, Direction.WEST):
                avail_actions[5] = 1

            # Can attack only alive units that are alive in the shooting range
            shoot_range = self.unit_shoot_range(unit)

            target_items = engine.enemies.items() if not is_opponent else engine.agents.items()
            self_items = engine.agents.items() if not is_opponent else engine.enemies.items()
            if engine.map_type == "MMM" and unit.unit_type == medivac_id:
                # Medivacs cannot heal themselves or other flying units
                target_items = [(t_id, t_unit) for (t_id, t_unit) in self_items if t_unit.unit_type != medivac_id]

            for t_id, t_unit in target_items:
                if t_unit.health > 0:
                    dist = distance(unit.pos.x, unit.pos.y, t_unit.pos.x, t_unit.pos.y)
                    if dist <= shoot_range:
                        if engine.map_type == "infestor_viper":
                            value = 0
                            # viper
                            if unit.unit_type == 499:
                                if unit.energy >= 125:
                                    value = 1
                            # infestor
                            else:
                                if unit.energy >= 50:
                                    value = 1
                            avail_actions[t_id + self.n_actions_no_attack] = value
                        else:
                            avail_actions[t_id + self.n_actions_no_attack] = 1

        else:
            # only no-op allowed
            avail_actions = [1] + [0] * (self.n_actions - 1)

        if (not skip_mirror) and self.mirror_opponent and is_opponent:
            avail_actions[MOVE_EAST], avail_actions[MOVE_WEST] = \
                avail_actions[MOVE_WEST], avail_actions[MOVE_EAST]

        return avail_actions

    def can_move(self, unit, direction):
        """Whether a unit can move in a given direction."""
        m = self._move_amount / 2

        if direction == Direction.NORTH:
            x, y = int(unit.pos.x), int(unit.pos.y + m)
        elif direction == Direction.SOUTH:
            x, y = int(unit.pos.x), int(unit.pos.y - m)
        elif direction == Direction.EAST:
            x, y = int(unit.pos.x + m), int(unit.pos.y)
        else:
            x, y = int(unit.pos.x - m), int(unit.pos.y)

        if self.check_bounds(x, y) and self.pathing_grid[x, y]:
            return True

        return False

    def check_bounds(self, x, y):
        """Whether a point is within the map bounds."""
        return 0 <= x < self.map_x and 0 <= y < self.map_y

    def get_surrounding_pathing(self, unit):
        """Returns pathing values of the grid surrounding the given unit."""
        points = self.get_surrounding_points(unit, include_self=False)
        vals = [self.pathing_grid[x, y] if self.check_bounds(x, y) else 1 for x, y in points]
        return vals

    def get_surrounding_height(self, unit):
        """Returns height values of the grid surrounding the given unit."""
        points = self.get_surrounding_points(unit, include_self=True)
        vals = [self.terrain_height[x, y] if self.check_bounds(x, y) else 1 for x, y in points]
        return vals

    def unit_shoot_range(self, unit):
        """Returns the shooting range for an agent."""
        type_id = unit.unit_type
        if type_id == 499:
            return 8
        elif type_id == 111:
            return 10
        else:
            return 6

    def get_surrounding_points(self, unit, include_self=False):
        """Returns the surrounding points of the unit in 8 directions."""
        x = int(unit.pos.x)
        y = int(unit.pos.y)

        ma = self._move_amount

        points = [
            (x, y + 2 * ma),
            (x, y - 2 * ma),
            (x + 2 * ma, y),
            (x - 2 * ma, y),
            (x + ma, y + ma),
            (x - ma, y - ma),
            (x + ma, y - ma),
            (x - ma, y + ma),
        ]

        if include_self:
            points.append((x, y))

        return points

    def get_movement_features(self, agent_id, engine, is_opponent=False):
        unit = self.get_unit_by_id(agent_id, engine, is_opponent=is_opponent)
        move_feats_dim = self.get_obs_move_feats_size()
        move_feats = np.zeros(move_feats_dim, dtype=np.float32)

        if unit.health > 0:  # otherwise dead, return all zeros
            # Movement features
            avail_actions = self.get_avail_agent_actions(agent_id, engine, is_opponent=is_opponent)
            for m in range(self.n_actions_move):
                move_feats[m] = avail_actions[m + 2]

            ind = self.n_actions_move

            if self.obs_pathing_grid:
                move_feats[ind:ind + self.n_obs_pathing  # TODO self.n_obs_pathing ?
                           ] = self.get_surrounding_pathing(unit)
                ind += self.n_obs_pathing

            if self.obs_terrain_height:
                move_feats[ind:] = self.get_surrounding_height(unit)
        return move_feats

    def get_obs_move_feats_size(self):
        """Returns the size of the vector containing the agents's movement-related features."""
        move_feats = self.n_actions_move
        if self.obs_pathing_grid:
            move_feats += self.n_obs_pathing
        if self.obs_terrain_height:
            move_feats += self.n_obs_height

        return move_feats

    def get_last_action(self, is_opponent=False):
        if is_opponent:
            ret = self.last_action_opponent
            if self.mirror_opponent:
                ret[:, MOVE_EAST], ret[:, MOVE_WEST] = \
                    ret[:, MOVE_WEST].copy(), ret[:, MOVE_EAST].copy()
        else:
            ret = self.last_action
        return ret

    def get_avail_actions(self, engine, is_opponent=False):
        return [
            self.get_avail_agent_actions(agent_id, engine, is_opponent=is_opponent)
            for agent_id in range(self.n_agents if not is_opponent else self.n_enemies)
        ]

    @staticmethod
    def _transform_action(a):
        if a == MOVE_EAST:  # intend to move east
            a = MOVE_WEST
        elif a == MOVE_WEST:  # intend to move west
            a = MOVE_EAST
        return a

    def info(self):
        shape = (self.n_actions, )
        value = {'min': 0, 'max': 1}
        return SMACAction.info_template(shape, value, None, None)
