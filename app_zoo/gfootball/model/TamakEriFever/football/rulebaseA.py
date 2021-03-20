
# https://www.kaggle.com/eugenkeil/simple-baseline-bot

from football.util import *


def _agent(obs):
    directions = [
    [Action.TopLeft, Action.Top, Action.TopRight],
    [Action.Left, Action.Idle, Action.Right],
    [Action.BottomLeft, Action.Bottom, Action.BottomRight]]

    dirsign = lambda x: 1 if abs(x) < 0.01 else (0 if x < 0 else 2)

    enemyGoal = [1, 0]
    perfectRange = [[0.6, 1], [-0.2, 0.2]]

    def inside(pos, area):
        return area[0][0] <= pos[0] <= area[0][1] and area[1][0] <= pos[1] <= area[1][1]

    obs = obs['players_raw'][0]
    controlled_player_pos = obs['left_team'][obs['active']]

    if obs["game_mode"] == GameMode.Penalty:
            return Action.Shot
    if obs["game_mode"] == GameMode.Corner:
        if controlled_player_pos[0] > 0:
            return Action.Shot
    if obs["game_mode"] == GameMode.FreeKick:
        return Action.Shot
    # Make sure player is running.
    if  0 < controlled_player_pos[0] < 0.6 and Action.Sprint not in obs['sticky_actions']:
        return Action.Sprint
    elif 0.6 < controlled_player_pos[0] and Action.Sprint in obs['sticky_actions']:
        return Action.ReleaseSprint

    # Does the player we control have the ball?
    if obs['ball_owned_player'] == obs['active'] and obs['ball_owned_team'] == 0:
        goalkeeper = 0
        if inside(controlled_player_pos, perfectRange) and controlled_player_pos[0] < obs['ball'][0]:
            return Action.Shot
        elif abs(obs['right_team'][goalkeeper][0] - 1) > 0.2 and controlled_player_pos[0] > 0.4 and abs(controlled_player_pos[1]) < 0.2:
            return Action.Shot
        else:
            xdir = dirsign(enemyGoal[0] - controlled_player_pos[0])
            ydir = dirsign(enemyGoal[1] - controlled_player_pos[1])
            return directions[ydir][xdir]
    else:
        # Run towards the ball.
        xdir = dirsign(obs['ball'][0] - controlled_player_pos[0])
        ydir = dirsign(obs['ball'][1] - controlled_player_pos[1])
        return directions[ydir][xdir]

def agent(obs):
    return [_agent(obs)]

