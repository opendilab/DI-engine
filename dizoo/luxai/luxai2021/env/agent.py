import sys
import time

import numpy as np
from gym import spaces
from ..game.constants import Constants

"""
Implements the base class for a training Agent
"""
class Agent:
    def __init__(self) -> None:
        """
        Implements an agent opponent
        """
        self.team = None
        self.match_controller = None
    
    def game_start(self, game):
        """
        This function is called at the start of each game. Use this to
        reset and initialize per game. Note that self.team may have
        been changed since last game. The game map has been created
        and starting units placed.

        Args:
            game ([type]): Game.
        """
        pass

    def process_turn(self, game, team):
        """
        Decides on a set of actions for the current turn.
        :param game:
        :param team:
        :return: Array of actions to perform for this turn.
        """
        actions = []
        return actions

    def pre_turn(self, game, is_first_turn=False):
        """
        Called before a turn starts. Allows for modifying the game environment.
        Generally only used in kaggle submission opponents.
        :param game:
        """
        return

    def post_turn(self, game, actions):
        """
        Called after a turn. Generally only used in kaggle submission opponents.
        :param game:
        :param actions:
        :return: (bool) True if it handled the turn (don't run our game engine)
        """
        return False

    def turn_heurstics(self, game, is_first_turn):
        """
        This is called pre-observation actions to allow for hardcoded heuristics
        to control a subset of units. Any unit or city that gets an action from this
        callback, will not create an observation+action.

        Args:
            game ([type]): Game in progress
            is_first_turn (bool): True if it's the first turn of a game.
        """
        return

    def get_agent_type(self):
        """
        Returns the type of agent. Use AGENT for inference, and LEARNING for training a model.
        """
        return Constants.AGENT_TYPE.AGENT

    def set_team(self, team):
        """
        Sets the team id that this agent is controlling
        :param team:
        """
        self.team = team

    def set_controller(self, match_controller):
        """
        """
        self.match_controller = match_controller


class AgentFromReplay(Agent):
    """
    Base class for an agent from a specified json replay file.
    """
    def __init__(self, replay=None) -> None:
        """
        Implements an agent opponent
        """
        super().__init__()
        self.replay = replay
    
    def get_agent_type(self):
        """
        Returns the type of agent. Use AGENT for inference, and LEARNING for training a model.
        """
        return Constants.AGENT_TYPE.AGENT

    def process_turn(self, game, team):
        """
        Decides on a set of actions for the current turn.
        :param game:
        :param team:
        :return: Array of actions to perform for this turn.
        """
        actions = []
        turn = game.state["turn"]

        if self.replay is not None:
            acts = self.replay['steps'][turn+1][team]["action"]
            acts = [game.action_from_string(a, team) for a in acts]
            acts = [a for a in acts if a is not None]
            actions.extend(acts)
        
        return actions
        
    

class AgentWithModel(Agent):
    """
    Base class for a stable_baselines3 reinforcement learning agent.
    """
    def __init__(self, mode="train", model=None) -> None:
        """
        Implements an agent opponent
        """
        super().__init__()
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(low=0, high=1, shape=(10,1), dtype=np.float16)

        self.model = model
        self.mode = mode
    
    def get_agent_type(self):
        """
        Returns the type of agent. Use AGENT for inference, and LEARNING for training a model.
        """
        if self.mode == "train":
            return Constants.AGENT_TYPE.LEARNING
        else:
            return Constants.AGENT_TYPE.AGENT
    
    def get_reward(self, game, is_game_finished, is_new_turn, is_game_error):
        """
        Returns the reward function for this step of the game. Reward should be a
        delta increment to the reward, not the total current reward.
        """
        return 0
    
    def get_observation(self, game, unit, city_tile, team, is_new_turn):
        """
        Implements getting a observation from the current game for this unit or city
        """
        return np.zeros((10,1))

    def process_turn(self, game, team):
        """
        Decides on a set of actions for the current turn. Not used in training, only inference. Generally
        don't modify this part of the code.
        Returns: Array of actions to perform.
        """
        start_time = time.time()
        actions = []
        new_turn = True

        # Inference the model per-unit
        units = game.state["teamStates"][team]["units"].values()
        for unit in units:
            if unit.can_act():
                obs = self.get_observation(game, unit, None, unit.team, new_turn)
                # IMPORTANT: You can change deterministic=True to disable randomness in model inference. Generally,
                # I've found the agents get stuck sometimes if they are fully deterministic.
                action_code, _states = self.model.predict(obs, deterministic=False)
                if action_code is not None:
                    actions.append(
                        self.action_code_to_action(action_code, game=game, unit=unit, city_tile=None, team=unit.team))
                new_turn = False

        # Inference the model per-city
        cities = game.cities.values()
        for city in cities:
            if city.team == team:
                for cell in city.city_cells:
                    city_tile = cell.city_tile
                    if city_tile.can_act():
                        obs = self.get_observation(game, None, city_tile, city.team, new_turn)
                        # IMPORTANT: You can change deterministic=True to disable randomness in model inference. Generally,
                        # I've found the agents get stuck sometimes if they are fully deterministic.
                        action_code, _states = self.model.predict(obs, deterministic=False)
                        if action_code is not None:
                            actions.append(
                                self.action_code_to_action(action_code, game=game, unit=None, city_tile=city_tile,
                                                           team=city.team))
                        new_turn = False

        time_taken = time.time() - start_time
        if time_taken > 0.5:  # Warn if larger than 0.5 seconds.
            print("WARNING: Inference took %.3f seconds for computing actions. Limit is 1 second." % time_taken,
                  file=sys.stderr)

        return actions


class AgentFromStdInOut(Agent):
    """
    Wrapper for an external agent where this agent's commands are coming in through standard input.
    """
    def __init__(self) -> None:
        """
        Implements an agent opponent
        """
        super().__init__()
        self.initialized_player = False
        self.initialized_map = False

    def pre_turn(self, game, is_first_turn=False):
        """
        Called before a turn starts. Allows for modifying the game environment.
        Generally only used in kaggle submission opponents.
        :param game:
        """

        # Read StdIn to update game state
        # Loosly implements:
        #    /Lux-AI-Challenge/Lux-Design-2021/blob/master/kits/python/simple/main.py
        #    AND /kits/python/simple/agent.py agent(observation, configuration)
        updates = []
        while True:
            message = input()

            if not self.initialized_player:
                team = int(message)
                self.set_team((team + 1) % 2)
                self.match_controller.set_opponent_team(self, team)

                self.initialized_player = True

            elif not self.initialized_map:
                # Parse the map size update message, it's always the second message of the game
                map_info = message.split(" ")
                game.configs["width"] = int(map_info[0])
                game.configs["height"] = int(map_info[1])

                # Use an empty map, because the updates will fill the map out
                game.configs["mapType"] = Constants.MAP_TYPES.EMPTY

                self.initialized_map = True
            else:
                updates.append(message)

            if message == "D_DONE":  # End of turn data marker
                break
        
        # Reset the game to the specified state. Don't increment turn counter on first turn of game.
        game.reset(updates=updates, increment_turn=not is_first_turn)

    def post_turn(self, game, actions) -> bool:
        """
        Called after a turn. Generally only used in kaggle submission opponents.
        :param game:
        :param actions:
        :return: (bool) True if it handled the turn (don't run our game engine)
        """
        # TODO: Send the list of actions to stdout in the correct format.
        messages = []
        for action in actions:
            messages.append(action.to_message(game))

        # Print the messages to the kaggle controller
        if len(messages) > 0:
            print(",".join(messages))
        else:
            # Print a new line. This is needed for the main_kaggle_submission.py wrapper to work
            print("")

        print("D_FINISH")

        # True here instructs the controller to not simulate the actions. Instead the kaggle controller will
        # run the turn and send back pre-turn map state.
        return True
