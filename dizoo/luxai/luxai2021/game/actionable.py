"""
Implements /src/Actionable/index.ts
"""


class Actionable:
    """
    Enum implementation
    """

    class Types:
        WOOD = 'wood'
        COAL = 'coal'
        URANIUM = 'uranium'

    def __init__(self, configs, cooldown=0.0) -> None:
        """

        :param configs:
        :param cooldown:
        """
        self.configs = configs
        self.current_actions = []
        self.cooldown = cooldown
        self.can_act_override = None

    def can_act(self) -> bool:
        """
        whether or not the unit can move or not.
        """
        if self.can_act_override == None:
            return self.cooldown < 1
        else:
            return self.can_act_override
    
    def set_can_act_override(self, can_act_override):
        """
        Override to whether this unit can act this turn.

        Args:
            can_act_override: True: Override that unit can act this turn. False: Override unit can't act this turn. None: No override.
        """
        self.can_act_override = can_act_override

    def handle_turn(self, game):
        """

        :param game:
        :return:
        """
        try:
            # ToDo self.turn() is not implemented
            self.turn(game)
        finally:
            self.current_actions = []
        # reset actions to empty

    def give_action(self, action):
        """

        :param action:
        :return:
        """
        self.current_actions.append(action)
