"""
Implements /src/Actions/index.ts
"""
from .constants import Constants

UNIT_TYPES = Constants.UNIT_TYPES


class Action:
    def __init__(self, action, team):
        self.action = action
        self.team = team

    def is_valid(self, game, actions_validated, accumulated_stats=None):
        """
        Validates the command.
        :param game:
        :param actions_validated: Other actions that have already been validated for this turn.
        :param accumulated_stats: List of accumulated stats of approved actions to help validate the action.
        :return: True if it's valid, False otherwise
        """
        return True

    def to_message(self, game):
        """
        Converts this action into a text message to send the kaggle controller via StdOut
        :param game:
        :return: String-serialized action message to send kaggle controller
        """
        raise Exception("NOT IMPLEMENTED")
    
    def commit_action_update_stats(self, game, accumulated_stats):
        """
        Updates the accumulated_stats with this action having been
        approved. Used to validate actions that depend on this, eg
        unit cap being reached from producing workers.

        Args:
            accumulated_stats ([Dict]): 
        """
        return accumulated_stats


class MoveAction(Action):
    def __init__(self, team, unit_id, direction, **kwarg):
        """

        :param team:
        :param unit_id:
        :param direction:
        :param kwarg:
        """
        action = Constants.ACTIONS.MOVE
        self.unit_id = unit_id
        self.direction = direction
        super().__init__(action, team)

    def is_valid(self, game, actions_validated, accumulated_stats=None):
        """
        Validates the command.
        :param game:
        :param actions_validated: Other actions that have already been validated for this turn.
        :return: True if it's valid, False otherwise
        """
        if self.unit_id is None or self.team is None or self.direction is None:
            return False

        unit = game.get_unit(self.team, self.unit_id)

        # Validate it can act
        if not unit.can_act():
            return False

        # Check map bounds of destination spot
        new_pos = unit.pos.translate(self.direction, 1)
        if new_pos.y < 0 or new_pos.y >= game.map.height:
            return False
        if new_pos.x < 0 or new_pos.x >= game.map.width:
            return False

        # Basic unit collision check
        target_cell = game.map.get_cell_by_pos(new_pos)
        if target_cell.is_city_tile() and target_cell.city_tile.team != unit.team:
            # collision with opponent city tile
            return False

        if not target_cell.is_city_tile():
            # Get units adjacent to target. Ignore opponents, because they might move.
            adjacent_cells = game.map.get_adjacent_cells(target_cell)
            adjacent_cells.append(target_cell) # Also look at the target cell

            # Index move actions
            moves = {}
            for action in actions_validated:
                if action.action == Constants.ACTIONS.MOVE:
                    moves[action.unit_id] = game.get_unit(action.team, action.unit_id).pos.translate(action.direction, 1)

            # Get potential collision units from our team
            for c in adjacent_cells:
                for id, u in c.units.items():
                    if u.team == self.team:
                        if id != self.unit_id:
                            # This unit is a potential collision
                            if id in moves:
                                # Check the unit move target for collision
                                if new_pos == moves[id]:
                                    # Collides with move target of existing unit
                                    return False
                            
                            # Check this unit's current position for a collision
                            if new_pos == unit.pos:
                                # Collides with a unit in the way
                                return False

        # Note: True collisions are handled in the turn loop as both players move
        return True

    def to_message(self, game) -> str:
        """
        Converts this action into a text message to send the kaggle controller via StdOut
        :param game:
        :return: (str) String-serialized action message to send kaggle controller
        """
        return "m {} {}".format(self.unit_id, self.direction)


class SpawnAction(Action):
    def __init__(self, action, team, unit_id, x, y, **kwarg):
        """
        
        :param action: 
        :param team: 
        :param unit_id: 
        :param x: 
        :param y: 
        :param kwarg: 
        """
        self.unit_id = unit_id
        self.x = x
        self.y = y
        super().__init__(action, team)

    def commit_action_update_stats(self, game, accumulated_stats):
        """
        Updates the accumulated_stats with this action having been
        approved. Used to validate actions that depend on this, eg
        unit cap being reached from producing workers.

        Args:
            accumulated_stats ([Dict]): 
        """
        if "unit_count_offset" in accumulated_stats[self.team]:
            accumulated_stats[self.team]["unit_count_offset"] += 1
        else:
            accumulated_stats[self.team]["unit_count_offset"] = 1
        
        return accumulated_stats
    
    def is_valid(self, game, actions_validated, accumulated_stats=None):
        """
        Validates the command.
        :param game:
        :param actions_validated: Other actions that have already been validated for this turn.
        :param accumulated_stats: List of accumulated stats of approved actions to help validate the action.
        :return: True if it's valid, False otherwise
        """
        if self.x is None or self.y is None or self.team is None:
            return False

        if self.unit_id != None:
            return False

        if self.y < 0 or self.y >= game.map.height:
            return False
        if self.x < 0 or self.x >= game.map.width:
            return False

        city_tile = game.map.get_cell(self.x, self.y).city_tile
        if city_tile is None:
            return False

        if not city_tile.can_build_unit():
            return False

        # Handles multiple cities building workers in same turn
        offset = 0
        if accumulated_stats:
            if "unit_count_offset" in accumulated_stats[self.team]:
                offset = accumulated_stats[self.team]["unit_count_offset"]
        
        if game.worker_unit_cap_reached(self.team, offset=offset):
            return False

        return True


class SpawnCartAction(SpawnAction):
    def __init__(self, team, unit_id, x, y, **kwarg):
        """
        
        :param team: 
        :param unit_id: 
        :param x: 
        :param y: 
        :param kwarg: 
        """
        action = Constants.ACTIONS.BUILD_CART
        self.type = UNIT_TYPES.CART
        super().__init__(action, team, unit_id, x, y)


    def to_message(self, game) -> str:
        """
        Converts this action into a text message to send the kaggle controller via StdOut
        :param game: 
        :return: (str) String-serialized action message to send kaggle controller
        """
        return "bc {} {}".format(self.x, self.y)


class SpawnWorkerAction(SpawnAction):
    def __init__(self, team, unit_id, x, y, **kwarg):
        """
        
        :param team: 
        :param unit_id: 
        :param x: 
        :param y: 
        :param kwarg: 
        """
        action = Constants.ACTIONS.BUILD_WORKER
        self.type = UNIT_TYPES.WORKER
        super().__init__(action, team, unit_id, x, y)

    def to_message(self, game) -> str:
        """
        Converts this action into a text message to send the kaggle controller via StdOut
        :param game:
        :return: (str) String-serialized action message to send kaggle controller
        """
        return "bw {} {}".format(self.x, self.y)


class SpawnCityAction(Action):
    def __init__(self, team, unit_id, **kwarg):
        """

        :param team:
        :param unit_id:
        :param kwarg:
        """
        action = Constants.ACTIONS.BUILD_CITY
        self.unit_id = unit_id
        super().__init__(action, team)

    def is_valid(self, game, actions_validated, accumulated_stats=None):
        """
        Validates the command.
        :param game:
        :param actions_validated: Other actions that have already been validated for this turn.
        :param accumulated_stats: List of accumulated stats of approved actions to help validate the action.
        :return: True if it's valid, False otherwise
        """
        if self.unit_id is None or self.team is None:
            return False

        unit = game.get_unit(self.team, self.unit_id)

        # Validate it can act
        if not unit.can_act():
            return False

        if not unit.can_build(game.map):
            return False

        # Validate the cell
        cell = game.map.get_cell_by_pos(unit.pos)
        if cell.is_city_tile():
            return False

        if cell.has_resource():
            return False

        # Note: Collisions are handled in the turn loop as both players move
        return True

    def to_message(self, game) -> str:
        """
        Converts this action into a text message to send the kaggle controller via StdOut
        :param game:
        :return: (str) String-serialized action message to send kaggle controller
        """
        return "bcity {}".format(self.unit_id)


class TransferAction(Action):
    def __init__(self, team, source_id, destination_id, resource_type, amount, **kwarg):
        """

        :param team:
        :param source_id:
        :param destination_id:
        :param resource_type:
        :param amount:
        :param kwarg:
        """
        action = Constants.ACTIONS.TRANSFER
        self.source_id = source_id
        self.destination_id = destination_id
        self.resource_type = resource_type
        self.amount = amount
        super().__init__(action, team)

    def to_message(self, game):
        """
        Converts this action into a text message to send the
        kaggle controller via StdOut
        Returns: String-serialized action message to send kaggle controller
        """
        return "t {} {} {} {}".format(self.source_id, self.destination_id, self.resource_type, self.amount)
    
    def is_valid(self, game, actions_validated, accumulated_stats=None):
        """
        Validates the command.
        :param game:
        :param actions_validated: Other actions that have already been validated for this turn.
        :param accumulated_stats: List of accumulated stats of approved actions to help validate the action.
        :return: True if it's valid, False otherwise
        """
        if self.source_id is None or self.destination_id is None or self.team is None or self.resource_type is None:
            return False

        if self.source_id == self.destination_id:
            return False

        unit_src = game.get_unit(self.team, self.source_id)
        unit_dst = game.get_unit(self.team, self.destination_id)

        if not unit_src.can_act():
            return False
        
        if not unit_src.pos.is_adjacent(unit_dst.pos):
            return False
        
        return True


class PillageAction(Action):
    def __init__(self, team, unit_id, **kwarg):
        """
        
        :param team: 
        :param unit_id: 
        :param kwarg:
        """
        action = Constants.ACTIONS.PILLAGE
        self.unit_id = unit_id
        super().__init__(action, team)

    def to_message(self, game) -> str:
        """
        Converts this action into a text message to send the kaggle controller via StdOut
        :param game:
        :return: (str) String-serialized action message to send kaggle controller
        """
        return "p {}".format(self.unit_id)
    
    def is_valid(self, game, actions_validated, accumulated_stats=None):
        """
        Validates the command.
        :param game:
        :param actions_validated: Other actions that have already been validated for this turn.
        :param accumulated_stats: List of accumulated stats of approved actions to help validate the action.
        :return: True if it's valid, False otherwise
        """
        if self.unit_id is None or self.team is None:
            return False

        unit = game.get_unit(self.team, self.unit_id)

        # Validate it can act
        if not unit.can_act():
            return False

        return True


class ResearchAction(Action):
    def __init__(self, team, x, y, unit_id, **kwarg):
        """
        Create a research action.

        Args:
            team ([type]):
            x ([type]):
            y ([type]):
        """
        action = Constants.ACTIONS.RESEARCH
        self.x = x
        self.y = y
        self.unit_id = unit_id
        super().__init__(action, team)

    def to_message(self, game) -> str:
        """
        Converts this action into a text message to send the kaggle controller via StdOut
        :param game:
        :return: (str) String-serialized action message to send kaggle controller
        """
        return "r {} {}".format(self.x, self.y)
    
    def is_valid(self, game, actions_validated, accumulated_stats=None):
        """
        Validates the command.
        :param game:
        :param actions_validated: Other actions that have already been validated for this turn.
        :param accumulated_stats: List of accumulated stats of approved actions to help validate the action.
        :return: True if it's valid, False otherwise
        """
        if self.x is None or self.y is None or self.team is None:
            return False
        
        if self.unit_id != None:
            return False

        if self.y < 0 or self.y >= game.map.height:
            return False
        if self.x < 0 or self.x >= game.map.width:
            return False

        city_tile = game.map.get_cell(self.x, self.y).city_tile
        if city_tile is None:
            return False

        if not city_tile.can_research():
            return False

        return True
