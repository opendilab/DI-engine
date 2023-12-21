import numpy as np
import torch
from gym import Env
from typing import Tuple, List


def get_vi_sequence(env: Env, observation: np.ndarray) -> Tuple[np.ndarray, List]:
    """
    Overview:
        Given an instance of the maze environment and the current observation, using Broad-First-Search (BFS) \
        algorithm to plan an optimal path and record the result.
    Arguments:
        - env (:obj:`Env`): The instance of the maze environment.
        - observation (:obj:`np.ndarray`): The current observation.
    Returns:
        - output (:obj:`Tuple[np.ndarray, List]`): The BFS result. ``output[0]`` contains the BFS map after each \
            iteration and ``output[1]`` contains the optimal actions before reaching the finishing point.
    """
    xy = np.where(observation[Ellipsis, -1] == 1)
    start_x, start_y = xy[0][0], xy[1][0]
    target_location = env.target_location
    nav_map = env.nav_map
    current_points = [target_location]
    chosen_actions = {target_location: 0}
    visited_points = {target_location: True}
    vi_sequence = []

    vi_map = np.full((env.size, env.size), fill_value=env.n_action, dtype=np.int32)

    found_start = False
    while current_points and not found_start:
        next_points = []
        for point_x, point_y in current_points:
            for (action, (next_point_x, next_point_y)) in [(0, (point_x - 1, point_y)), (1, (point_x, point_y - 1)),
                                                           (2, (point_x + 1, point_y)), (3, (point_x, point_y + 1))]:

                if (next_point_x, next_point_y) in visited_points:
                    continue

                if not (0 <= next_point_x < len(nav_map) and 0 <= next_point_y < len(nav_map[next_point_x])):
                    continue

                if nav_map[next_point_x][next_point_y] == 'x':
                    continue

                next_points.append((next_point_x, next_point_y))
                visited_points[(next_point_x, next_point_y)] = True
                chosen_actions[(next_point_x, next_point_y)] = action
                vi_map[next_point_x, next_point_y] = action

                if next_point_x == start_x and next_point_y == start_y:
                    found_start = True
        vi_sequence.append(vi_map.copy())
        current_points = next_points
    track_back = []
    if found_start:
        cur_x, cur_y = start_x, start_y
        while cur_x != target_location[0] or cur_y != target_location[1]:
            act = vi_sequence[-1][cur_x, cur_y]
            track_back.append((torch.FloatTensor(env.process_states([cur_x, cur_y], env.get_maze_map())), act))
            if act == 0:
                cur_x += 1
            elif act == 1:
                cur_y += 1
            elif act == 2:
                cur_x -= 1
            elif act == 3:
                cur_y -= 1

    return np.array(vi_sequence), track_back
