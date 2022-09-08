# encoding: utf-8

from collections import defaultdict
import logging
import numpy as np


class GomokuExpert(object):
    def __init__(self):
        self.grade = 100
        # The weight of pieces of the same color
        self.max_value = 10012345
        # Indicates that it has already reached 5, reaching the maximum weight
        self.init_board_flag = False

    def location_to_move(self, i, j):
        # location = (i,j),move=j+i*width
        return j + i * self.board_width

    def move_to_location(self, move):
        # location = (i,j),dot=j+i*width

        j = move % self.board_width
        i = move // self.board_width
        return [i, j]

    def get_loc_player(self, i, j):
        move = self.location_to_move(i, j)
        return self.board_status[move]

    def scan_leftright(self, i, j, player):
        value = 0
        count = 0
        # Count the number of consecutive own flags or empty chess pieces
        grade = self.grade

        m, n = i, j - 1
        # move one step to the left
        while n >= 0:
            # Incoming move position and player
            is_continue, value, grade = self.caculate_once_value(m, n,
                                                                 player,
                                                                 value,
                                                                 grade)
            if is_continue:
                n = n - 1
            else:
                break
                # move one step to the left
            count += 1

        grade = self.grade
        # Change the direction, the weights are returned to the initial values
        n = j + 1
        while n < self.board_width:
            is_continue, value, grade = self.caculate_once_value(m, n,
                                                                 player,
                                                                 value,
                                                                 grade)
            if is_continue:
                n = n + 1
            else:
                break
                # move one step to the right
            count += 1

        return value if count >= 4 else 0
        # If this direction can be connected into 5, return value, otherwise it is 0

    def scan_updown(self, i, j, player):
        value = 0
        count = 0
        # Count the number of consecutive own chess pieces or empty chess pieces
        grade = self.grade

        m, n = i - 1, j
        # move one step up
        while m >= 0:
            is_continue, value, grade = self.caculate_once_value(m, n,
                                                                 player,
                                                                 value,
                                                                 grade)
            if is_continue:
                m = m - 1
            else:
                break
                # move one step up
            count += 1

        grade = self.grade
        # Change the direction and change the weight back to the initial value
        m = i + 1
        # down
        while m < self.board_height:
            is_continue, value, grade = self.caculate_once_value(m, n,
                                                                 player,
                                                                 value,
                                                                 grade)
            if is_continue:
                m = m + 1
            else:
                break
                # move down one step
            count += 1

        return value if count >= 4 else 0
        # If this direction can be connected into 5, return value, otherwise it is 0

    def scan_left_updown(self, i, j, player):
        value = 0
        count = 0
        # Count the number of consecutive own chess pieces or empty chess pieces

        grade = self.grade
        m, n = i - 1, j - 1
        # Move up one step to the left
        while m >= 0 and n >= 0:
            is_continue, value, grade = self.caculate_once_value(m, n,
                                                                 player,
                                                                 value,
                                                                 grade)
            if is_continue:
                m, n = m - 1, n - 1
            else:
                break
                # Move up one step to the left
            count += 1

        grade = self.grade
        m, n = i + 1, j + 1
        # down right
        while m < self.board_height and n < self.board_width:
            is_continue, value, grade = self.caculate_once_value(m, n,
                                                                 player,
                                                                 value,
                                                                 grade)
            if is_continue:
                m, n = m + 1, n + 1
            else:
                break
                # Continue to move one step down to the right
            count += 1

        return value if count >= 4 else 0
        # If this direction can be connected into 5, return value, otherwise it is 0

    def scan_right_updown(self, i, j, player):
        value = 0
        count = 0
        # Count the number of consecutive own chess pieces or empty chess pieces

        grade = self.grade
        m, n = i + 1, j - 1
        # Move down one step to the left
        while m < self.board_height and n >= 0:
            is_continue, value, grade = self.caculate_once_value(m, n,
                                                                 player,
                                                                 value,
                                                                 grade)
            if is_continue:
                m, n = m + 1, n - 1
            else:
                break
                # Move down one step to the left
            count += 1
        grade = self.grade
        m, n = i - 1, j + 1
        # right up
        while m >= 0 and n < self.board_width:
            is_continue, value, grade = self.caculate_once_value(m, n,
                                                                 player,
                                                                 value,
                                                                 grade)
            if is_continue:
                m, n = m - 1, n + 1
            else:
                break
                # Continue to move up one step to the right
            count += 1

        return value if count >= 4 else 0
        # If this direction can be connected into 5, return value, otherwise it is 0

    def caculate_once_value(self, m, n, player, value, grade):
        # Calculate the value of the player at the specified position

        # return 0 1 2
        loc_player = self.get_loc_player(m, n)

        if loc_player == player:
            value += grade
        elif loc_player == 0:
            value += 1
            grade = grade / 10
            # When encountering an empty chess, reduce the weight of subsequent black chess
        else:
            # opponent's pawn
            value -= 5
            return 0, value, grade
            # When encountering an opponent's chess, end the follow-up move
        # 1 means 'is_continue'
        return 1, value, grade

    def evaluate_all_value(self, player):
        # Evaluate all vacancies for value

        self.move_value = defaultdict(lambda: [0, 0, 0, 0, 0])
        for move in self.available:
            i, j = self.move_to_location(move)

            self.move_value[move][0] = self.scan_updown(i, j, player)
            self.move_value[move][1] = self.scan_leftright(i, j, player)
            self.move_value[move][2] = self.scan_left_updown(i, j, player)
            self.move_value[move][3] = self.scan_right_updown(i, j, player)

            # Indicates that one direction can already be rushed to 4
            for k in range(4):
                if self.move_value[move][k] >= 390:
                    self.move_value[move][k] = 2000
                elif self.move_value[move][k] >= 302:
                    self.move_value[move][k] = 1000

            # Comprehensive score in all directions
            self.move_value[move][4] = (self.move_value[move][0] +
                                        self.move_value[move][1] +
                                        self.move_value[move][2] +
                                        self.move_value[move][3])

        move = max(self.available, key=lambda x: self.move_value[x][4])

        return move, self.move_value[move][4]

    def get_move(self, obs):
        self.board = obs

        if self.init_board_flag is False:
            self.board_width = self.board['observation'][0].shape[0]
            self.board_height = self.board['observation'][0].shape[1]
            # the 2 dim indicates which player is the to_play player, 1 means player 1, 2 means player 2
            if self.board['observation'][2][0][0] == 1:
                self.m_player_id = 1
                self.s_player_id = 2
            else:
                self.m_player_id = 2
                self.s_player_id = 1
            self.init_board_flag = True

        # transform observation,action_mask to self.available, self.board_status
        self.available = []
        self.board_status = np.zeros(self.board_width * self.board_height, 'int8')
        for i in range(self.board_width):
            for j in range(self.board_height):
                move = self.location_to_move(i, j)
                if self.board['action_mask'][move] == 1:
                    self.available.append(move)
                if self.board['observation'][0][i][j] == 1:
                    self.board_status[move] = self.m_player_id
                elif self.board['observation'][1][i][j] == 1:
                    self.board_status[move] = self.s_player_id

        m_move, m_value = self.evaluate_all_value(self.m_player_id)

        logging.info("loaction:{loc},value:{value}".format(
                     loc=self.move_to_location(m_move), value=m_value))

        s_move, s_value = self.evaluate_all_value(self.s_player_id)
        logging.info("O_loaction:{loc},value:{value}".format(
                      loc=self.move_to_location(s_move), value=s_value))
        # Block this position if it is better for the opponent to play here
        if m_value >= s_value:
            return m_move
        else:
            return s_move
