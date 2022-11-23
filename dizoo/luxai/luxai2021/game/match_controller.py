from luxai2021.game.city import CityTile
import random
import sys
import time
import traceback

from .constants import Constants
from ..env.agent import Agent


class GameStepFailedException(Exception):
    pass


class ActionSequence():
    def __init__(self, actions, unit_id, citytile, **kwarg):
        """
        Defines a sequence of actions, to be taken each time the unit or city can next move.

        Example usage of constructor:
            sequence = ActionSequence(
                actions=[
                    partial(MoveAction, direction=Constants.DIRECTIONS.NORTH),
                    partial(MoveAction, direction=Constants.DIRECTIONS.NORTH),
                    partial(MoveAction, direction=Constants.DIRECTIONS.NORTH),
                    partial(MoveAction, direction=Constants.DIRECTIONS.NORTH),
                    partial(MoveAction, direction=Constants.DIRECTIONS.NORTH),
                ],
                game=game,
                unit_id=unit.id if unit else None,
                unit=unit,
                city_id=city_tile.city_id if city_tile else None,
                citytile=city_tile,
                team=team,
                x=x,
                y=y
            )
            match_controller.take_action(sequence)
        
        Example usage as part of the template agent_policy.py action space:
            self.actionSpaceMap = [
                partial(MoveAction, direction=Constants.DIRECTIONS.CENTER),  # This is the do-nothing action
                partial(MoveAction, direction=Constants.DIRECTIONS.NORTH),
                partial(MoveAction, direction=Constants.DIRECTIONS.WEST),
                partial(MoveAction, direction=Constants.DIRECTIONS.SOUTH),
                partial(MoveAction, direction=Constants.DIRECTIONS.EAST),
                partial(ActionSequence, actions=[
                    partial(MoveAction, direction=Constants.DIRECTIONS.NORTH),
                    partial(MoveAction, direction=Constants.DIRECTIONS.NORTH),
                    partial(MoveAction, direction=Constants.DIRECTIONS.NORTH),
                    partial(MoveAction, direction=Constants.DIRECTIONS.NORTH),
                    partial(MoveAction, direction=Constants.DIRECTIONS.NORTH),
                ]),
                ...
            ]
        
        You can override this class with more complex logic, eg sequences
        like 'move to nearest city'
        """
        self.actions = list(actions)
        self.unit_id = unit_id
        self.citytile = citytile
        self.kwarg = kwarg
    
    def get_next_action(self, game):
        # Construct the next action. Note: x and y may be wrong since they may have changed
        # TODO: Fix x and y if doing citytile actions or build city action.
        return self.actions.pop(0)(unit_id=self.unit_id, citytile=self.citytile, **self.kwarg)
    
    def is_done(self):
        return len(self.actions) == 0


class MatchController:
    def __init__(self, game, agents=[None, None], replay_validate=None) -> None:
        """

        :param game:
        :param agents:
        """
        self.action_buffer = []
        self.game = game
        self.agents = agents
        self.replay_validate = replay_validate

        if len(agents) != 2:
            raise ValueError("Two agents must be specified.")

        # Validate the agents
        self.training_agent_count = 0
        for i, agent in enumerate(agents):
            if not (issubclass(type(agent), Agent) or isinstance(agent, Agent)):
                raise ValueError("All agents must inherit from Agent.")
            if agent.get_agent_type == Constants.AGENT_TYPE.LEARNING:
                self.training_agent_count += 1

            # Initialize agent
            agent.set_team(i)
            agent.set_controller(self)
        
        # Reset the agents, without resetting the game
        self.reset(reset_game=False)

        if self.training_agent_count > 1:
            raise ValueError("At most one agent must be trainable.")

        elif self.training_agent_count == 1:
            print("Running in training mode.", file=sys.stderr)

        elif self.training_agent_count == 0:
            print("Running in inference-only mode.", file=sys.stderr)

    def reset(self, reset_game=True, randomize_team_order=True):
        """

        :return:
        """
        # Randomly re-assign teams of the agents
        if randomize_team_order:
            r = random.randint(0, 1)
            self.agents[0].set_team(r)
            self.agents[1].set_team((r + 1) % 2)

        # Reset action sequences
        self.action_sequences = {}

        # Reset the game as well if needed
        if reset_game:
            self.game.reset()
        self.action_buffer = []
        self.accumulated_stats = dict( {Constants.TEAM.A: {}, Constants.TEAM.B: {}} )

        # Call the agent game_start() callbacks
        for agent in self.agents:
            agent.game_start(self.game)

    def take_action(self, action):
        """
         Adds the specified action to the action buffer
         """
        if action is not None:
            # Check if this is an action sequence
            if issubclass(type(action), ActionSequence) or isinstance(action, ActionSequence):
                # Add this action sequence and pop the first index off
                sequence = action
                if sequence.is_done():
                    return
                action = sequence.get_next_action(self.game)
                if action == None:
                    return

                if not sequence.is_done():
                    if sequence.unit_id != None:
                        self.action_sequences[sequence.unit_id] = sequence
                    elif sequence.citytile != None:
                        self.action_sequences[sequence.citytile] = sequence

            # Validate the action
            try:
                if action.is_valid(self.game, self.action_buffer, self.accumulated_stats):
                    # Add the action
                    self.action_buffer.append(action)
                    self.accumulated_stats = action.commit_action_update_stats(self.game, self.accumulated_stats)
                else:
                    #print(f'action is invalid {action} turn {self.game.state["turn"]}: {vars(action)}', file=sys.stderr)
                    pass
                    
            except KeyError:
                print(f'action failed, probably a dead unit {action}: {vars(action)}', file=sys.stderr)

        # Mark the unit or city as not able to perform another action this turn
        actionable = None
        if hasattr(action, 'unit_id') and action.unit_id is not None:
            # Mark the unit as already-actioned this turn
            if action.unit_id in self.game.state["teamStates"][0]["units"]:
                actionable = self.game.state["teamStates"][0]["units"][action.unit_id]
            elif action.unit_id in self.game.state["teamStates"][1]["units"]:
                actionable = self.game.state["teamStates"][1]["units"][action.unit_id]

        elif hasattr(action, 'x') and action.x is not None:
            # Mark the city as already-actioned this turn
            cell = self.game.map.get_cell(action.x, action.y)
            if cell.is_city_tile():
                actionable = cell.city_tile

        if actionable is not None:
            actionable.set_can_act_override(False)

    def take_actions(self, actions):
        """
         Adds the specified action to the action buffer
        """
        if actions != None:
            for action in actions:
                self.take_action(action)

    def log_error(self, text):
        # Ignore errors caused by logger
        try:
            if text is not None:
                with open("match_errors.txt", "a") as o:
                    o.write(text + "\n")
        except Exception:
            print("Critical error in logging")

    def set_opponent_team(self, agent, team):
        """
        Sets the team of the opposing team
        """
        for a in self.agents:
            if a != agent:
                a.set_team(team)

    def run_to_next_observation(self):
        """ 
            Generator function that gets the observation at the next Unit/City
            to be controlled.
            Returns: tuple describing the unit who's control decision is for (unit_id, city, team, is new turn)
        """
        game_over = False
        is_first_turn = True
        while not game_over:
            turn = self.game.state["turn"]

            # Run pre-turn agent events to allow for them to handle running the turn instead (used in a kaggle submission agent)
            for agent in self.agents:
                agent.pre_turn(self.game, is_first_turn)

            # Process any pending action sequences to automatically apply actions to units for this turn
            for id in list(self.action_sequences.keys()):
                sequence = self.action_sequences[id]
                actionable = None
                if id in self.game.state["teamStates"][0]["units"]:
                    actionable = self.game.state["teamStates"][0]["units"][id]
                elif id in self.game.state["teamStates"][1]["units"]:
                    actionable = self.game.state["teamStates"][1]["units"][id]
                elif isinstance(id, CityTile):
                    # Validate the city still exists
                    if id.city_id in self.game.cities:
                        actionable = id
                else:
                    # The unit must no longer exist
                    pass

                if actionable != None and actionable.can_act():
                    # Continue the action sequence for this unit automatically
                    self.take_action(sequence.get_next_action(self.game))

                    if sequence.is_done():
                        self.action_sequences.pop(id)
                elif actionable == None:
                    # Delete the action sequence, the object isn't valid anymore
                    self.action_sequences.pop(id)
            
            # Run agent.turn_heurstics() to apply any agent heristics to give units orders
            for agent in self.agents:
                agent.turn_heurstics(self.game, is_first_turn)

            # Process this turn
            for agent in self.agents:
                if agent.get_agent_type() == Constants.AGENT_TYPE.AGENT:
                    # Call the agent for the set of actions
                    actions = agent.process_turn(self.game, agent.team)
                    self.take_actions(actions)

                elif agent.get_agent_type() == Constants.AGENT_TYPE.LEARNING:
                    # Yield the game to make a decision, since the learning environment is the function caller
                    new_turn = True
                    start_time = time.time()

                    units = self.game.state["teamStates"][agent.team]["units"].values()
                    for unit in units:
                        if unit.can_act():
                            # RL training agent that is controlling the simulation
                            # The enviornment then handles this unit, and calls take_action() to buffer a requested action
                            yield unit, None, unit.team, new_turn
                            new_turn = False
                            

                    cities = self.game.cities.values()
                    for city in cities:
                        if city.team == agent.team:
                            for cell in city.city_cells:
                                city_tile = cell.city_tile
                                if city_tile.can_act():
                                    # RL training agent that is controlling the simulation
                                    # The enviornment then handles this city, and calls take_action() to buffer a requested action
                                    yield None, city_tile, city_tile.team, new_turn
                                    new_turn = False

                    time_taken = time.time() - start_time
            
            # Reset the can_act overrides for all units and city_tiles
            units = list(self.game.state["teamStates"][0]["units"].values()) + list(self.game.state["teamStates"][1]["units"].values())
            for unit in units:
                unit.set_can_act_override(None)
            for city in self.game.cities.values():
                for cell in city.city_cells:
                    city_tile = cell.city_tile.set_can_act_override(None)

            is_first_turn = False

            # Now let the game actually process the requested actions and play the turn
            try:
                # Run post-turn agent events to allow for them to handle running the turn instead (used in a kaggle submission agent)
                self.accumulated_stats = dict( {Constants.TEAM.A: {}, Constants.TEAM.B: {}} )
                handled = False
                for agent in self.agents:
                    if agent.post_turn(self.game, self.action_buffer):
                        handled = True

                if not handled:
                    game_over = self.game.run_turn_with_actions(self.action_buffer)
            except Exception as e:
                # Log exception
                self.log_error("ERROR: Critical error occurred in turn simulation.")
                self.log_error(repr(e))
                self.log_error(''.join(traceback.format_exception(None, e, e.__traceback__)))
                raise GameStepFailedException("Critical error occurred in turn simulation.")

            self.action_buffer = []

            if self.replay_validate is not None:
                self.game.process_updates(self.replay_validate['steps'][turn+1][0]['observation']['updates'], assign=False)
