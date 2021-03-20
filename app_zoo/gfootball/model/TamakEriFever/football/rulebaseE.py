
# https://www.kaggle.com/shotaro/optimizing-tunable-bot-with-optuna
## TUNABLE BASELINE BOT

# Tune Here:
SPRINT_RANGE = 0.6

SHOT_RANGE_X = 0.7
SHOT_RANGE_Y = 0.2

GOALIE_OUT = 0.2
LONG_SHOT_X = 0.4
LONG_SHOT_Y = 0.2

from football.util import *
from math import sqrt

directions = [
[Action.TopLeft, Action.Top, Action.TopRight],
[Action.Left, Action.Idle, Action.Right],
[Action.BottomLeft, Action.Bottom, Action.BottomRight]]

dirsign = lambda x: 1 if abs(x) < 0.01 else (0 if x < 0 else 2)

enemyGoal = [1, 0]
GOALKEEPER = 0

shot_range = [[SHOT_RANGE_X, 1],
                  [-SHOT_RANGE_Y, SHOT_RANGE_Y]]

def inside(pos, area):
    return area[0][0] <= pos[0] <= area[0][1] and area[1][0] <= pos[1] <= area[1][1]

def _agent(obs):
    return agent(obs)[0]

@human_readable_agent
def agent(obs):
    controlled_player_pos = obs['left_team'][obs['active']]

    if obs["game_mode"] == GameMode.Penalty:
        return Action.Shot
    if obs["game_mode"] == GameMode.Corner:
        if controlled_player_pos[0] > 0:
            return Action.Shot
    if obs["game_mode"] == GameMode.FreeKick:
        return Action.Shot

    # Make sure player is running down the field.
    if  0 < controlled_player_pos[0] < SPRINT_RANGE and Action.Sprint not in obs['sticky_actions']:
        return Action.Sprint
    elif SPRINT_RANGE < controlled_player_pos[0] and Action.Sprint in obs['sticky_actions']:
        return Action.ReleaseSprint

    # If our player controls the ball:
    if obs['ball_owned_player'] == obs['active'] and obs['ball_owned_team'] == 0:

        if inside(controlled_player_pos, shot_range) and controlled_player_pos[0] < obs['ball'][0]:
            return Action.Shot

        elif ( abs(obs['right_team'][GOALKEEPER][0] - 1) > GOALIE_OUT
                and controlled_player_pos[0] > LONG_SHOT_X and abs(controlled_player_pos[1]) < LONG_SHOT_Y ):
            return Action.Shot

        else:
            xdir = dirsign(enemyGoal[0] - controlled_player_pos[0])
            ydir = dirsign(enemyGoal[1] - controlled_player_pos[1])
            return directions[ydir][xdir]

    # if we we do not have the ball:
    else:
        # Run towards the ball.
        xdir = dirsign(obs['ball'][0] - controlled_player_pos[0])
        ydir = dirsign(obs['ball'][1] - controlled_player_pos[1])
        return directions[ydir][xdir]
