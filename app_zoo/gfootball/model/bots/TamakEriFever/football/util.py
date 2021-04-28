# https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/football/helpers.py

import enum
from functools import wraps
from typing import *


class Action(enum.IntEnum):
    Idle = 0
    Left = 1
    TopLeft = 2
    Top = 3
    TopRight = 4
    Right = 5
    BottomRight = 6
    Bottom = 7
    BottomLeft = 8
    LongPass = 9
    HighPass = 10
    ShortPass = 11
    Shot = 12
    Sprint = 13
    ReleaseDirection = 14
    ReleaseSprint = 15
    Slide = 16
    Dribble = 17
    ReleaseDribble = 18


sticky_index_to_action = [
    Action.Left, Action.TopLeft, Action.Top, Action.TopRight, Action.Right, Action.BottomRight, Action.Bottom,
    Action.BottomLeft, Action.Sprint, Action.Dribble
]

action_to_sticky_index = {a: index for index, a in enumerate(sticky_index_to_action)}


class PlayerRole(enum.IntEnum):
    GoalKeeper = 0
    CenterBack = 1
    LeftBack = 2
    RightBack = 3
    DefenceMidfield = 4
    CentralMidfield = 5
    LeftMidfield = 6
    RIghtMidfield = 7
    AttackMidfield = 8
    CentralFront = 9


class GameMode(enum.IntEnum):
    Normal = 0
    KickOff = 1
    GoalKick = 2
    FreeKick = 3
    Corner = 4
    ThrowIn = 5
    Penalty = 6


def human_readable_agent(agent: Callable[[Dict], Action]):
    """
    Decorator allowing for more human-friendly implementation of the agent function.
    @human_readable_agent
    def my_agent(obs):
        ...
        return football_action_set.action_right
    """

    @wraps(agent)
    def agent_wrapper(obs) -> List[int]:
        # Extract observations for the first (and only) player we control.
        obs = obs['players_raw'][0]
        # Turn 'sticky_actions' into a set of active actions (strongly typed).
        obs['sticky_actions'] = {
            sticky_index_to_action[nr]
            for nr, action in enumerate(obs['sticky_actions']) if action
        }
        # Turn 'game_mode' into an enum.
        obs['game_mode'] = GameMode(obs['game_mode'])
        # In case of single agent mode, 'designated' is always equal to 'active'.
        if 'designated' in obs:
            del obs['designated']
        # Conver players' roles to enum.
        obs['left_team_roles'] = [PlayerRole(role) for role in obs['left_team_roles']]
        obs['right_team_roles'] = [PlayerRole(role) for role in obs['right_team_roles']]

        action = agent(obs)
        return [action.value]

    return agent_wrapper
