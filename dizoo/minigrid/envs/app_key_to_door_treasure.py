#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gym_minigrid.minigrid import *
from gym_minigrid.minigrid import WorldObj


class Ball(WorldObj):

    def __init__(self, color='blue'):
        super(Ball, self).__init__('ball', color)

    def can_pickup(self):
        return False

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])


class AppleKeyToDoorTreasure(MiniGridEnv):
    """
    Classic 4 rooms gridworld environment.
    Can specify agent and goal position, if not it set at random.
    """

    def __init__(self, agent_pos=None, goal_pos=None, grid_size=19, apple=2):
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos
        self.apple = apple
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
        self.put_obj(Key('yellow'), *pos1)
        pos2_dummy_list = []  # to avoid overlap of apples
        for i in range(self.apple):
            pos2 = (self._rand_int(1, room_w), self._rand_int(1, room_h))
            while pos2 in pos2_dummy_list:
                pos2 = (self._rand_int(1, room_w), self._rand_int(1, room_h))
            self.put_obj(Ball('red'), *pos2)
            pos2_dummy_list.append(pos2)
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

    def _reward_ball(self):
        """
        Compute the reward to be given upon finding the apple
        """

        return 1

    def _reward_goal(self):
        """
        Compute the reward to be given upon success
        """

        return 10

    def step(self, action):
        self.step_count += 1

        reward = 0
        done = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell == None or fwd_cell.can_overlap():  # Ball and keys' can_overlap are False
                self.agent_pos = fwd_pos
            if fwd_cell != None and fwd_cell.type == 'goal':
                done = True
                reward = self._reward_goal()
            if fwd_cell != None and fwd_cell.type == 'ball':
                reward = self._reward_ball()
                self.grid.set(*fwd_pos, None)
                self.agent_pos = fwd_pos
            if fwd_cell != None and fwd_cell.type == 'lava':
                done = True

        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)

        # Drop an object
        elif action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(*fwd_pos, self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None

        # Toggle/activate an object: Here, this will open the door if you have the right key
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            assert False, "unknown action"

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()

        return obs, reward, done, {}


class AppleKeyToDoorTreasure_13x13(AppleKeyToDoorTreasure):

    def __init__(self):
        super().__init__(agent_pos=(2, 8), goal_pos=(7, 1), grid_size=13, apple=2)


class AppleKeyToDoorTreasure_19x19(AppleKeyToDoorTreasure):

    def __init__(self):
        super().__init__(agent_pos=(2, 14), goal_pos=(10, 1), grid_size=19, apple=2)


class AppleKeyToDoorTreasure_13x13_1(AppleKeyToDoorTreasure):

    def __init__(self):
        super().__init__(agent_pos=(2, 8), goal_pos=(7, 1), grid_size=13, apple=1)


class AppleKeyToDoorTreasure_7x7_1(AppleKeyToDoorTreasure):

    def __init__(self):
        super().__init__(agent_pos=(1, 5), goal_pos=(4, 1), grid_size=7, apple=1)


class AppleKeyToDoorTreasure_19x19_3(AppleKeyToDoorTreasure):

    def __init__(self):
        super().__init__(agent_pos=(2, 14), goal_pos=(10, 1), grid_size=19, apple=3)


if __name__ == '__main__':
    AppleKeyToDoorTreasure()._gen_grid(13, 13)  # Note that Minigrid has set seeds automatically
