#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gym_minigrid.minigrid import *
from gym_minigrid.minigrid import WorldObj
'''
class Apple(WorldObj):
    def __init__(self, color='red'):
        super(Apple, self).__init__('apple', color)

    def can_pickup(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])
'''


class AppleKeyToDoorTreasure(MiniGridEnv):
    """
    Classic 4 rooms gridworld environment.
    Can specify agent and goal position, if not it set at random.
    """

    def __init__(self, agent_pos=None, goal_pos=None, grid_size=19):
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos
        super().__init__(grid_size=grid_size, max_steps=100)

    def _gen_grid(
        self, width, height
    ):  # Note that it is inherited from MiniGridEnv that if width and height == None, width = grid_size , height = grid_size
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        room_w = width // 2
        room_h = height // 2

        # For each row of rooms
        for j in range(0, 2):

            # For each column
            for i in range(0, 2):
                xL = i * room_w
                yT = j * room_h
                xR = xL + room_w
                yB = yT + room_h

                # Bottom wall and door
                if i + 1 < 2:
                    if j + 1 < 2:
                        self.grid.vert_wall(xR, yT, room_h)
                        #pos = (xR, self._rand_int(yT + 1, yB))
                    else:
                        self.grid.vert_wall(xR, yT, room_h)
                        pos = (xR, self._rand_int(yT + 1, yB))
                        self.grid.set(*pos, None)

                # Bottom wall and door
                if j + 1 < 2:
                    if i + 1 < 2:
                        self.grid.horz_wall(xL, yB, room_w)
                        pos = (self._rand_int(xL + 1, xR), yB)
                        self.grid.set(*pos, None)
                    else:
                        self.grid.horz_wall(xL, yB, room_w)
                        pos = (self._rand_int(xL + 1, xR), yB)
                        self.put_obj(Door('yellow', is_locked=True), *pos)

        # Place a yellow key on the left side
        pos1 = (self._rand_int(room_w + 1, 2 * room_w), self._rand_int(room_h + 1, 2 * room_h))  # self._rand_int: [)
        pos2 = (self._rand_int(1, room_w), self._rand_int(1, room_h))
        self.place_obj(obj=Key('yellow'), top=pos1)
        self.place_obj(obj=Ball('red'), size=pos2)
        #self.place_obj(Key('yellow'),*pos)
        # Randomize the player start position and orientation
        if self._agent_default_pos is not None:
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)
            self.agent_dir = self._rand_int(0, 4)  # assuming random start direction
        else:
            self.place_agent()

        if self._goal_default_pos is not None:
            goal = Goal()
            self.put_obj(goal, *self._goal_default_pos)
            goal.init_pos, goal.cur_pos = self._goal_default_pos
        else:
            self.place_obj(Goal())

        self.mission = 'Reach the goal'

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)
        return obs, reward, done, info


class AppleKeyToDoorTreasure_13x13(AppleKeyToDoorTreasure):

    def __init__(self):
        super().__init__(agent_pos=(2, 8), goal_pos=(7, 1), grid_size=13)


class AppleKeyToDoorTreasure_19x19(AppleKeyToDoorTreasure):

    def __init__(self):
        super().__init__(agent_pos=(2, 14), goal_pos=(10, 1), grid_size=19)


#AppleKeyToDoorTreasure()._gen_grid(13,13)  #Note that Minigrid has set seeds automatically
