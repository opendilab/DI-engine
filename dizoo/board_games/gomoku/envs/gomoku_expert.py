# Reference link:
# https://github.com/LouisCaixuran/gomoku/blob/c1b6d508522d9e8c78be827f326bbee54c4dfd8b/gomoku/expert.py

from collections import defaultdict
import logging
import numpy as np


class GomokuExpert(object):
    """
        Overview:
            The ``GomokuExpert`` used to output rule-based gomoku expert actions. \
            Input is gomoku board obs(:obj:`np,array`) of shape ``(board_w, board_h)`` and returns action (:obj:`Int`) \
            The output action is the sequence number i*board_w+j of the placement position (i, j)
        Interfaces:
            ``__init__``, ``get_action``.
    """
    def __init__(self):
        """
        Overview:
            Init the ``GomokuExpert``.
        """
        # The weight of pieces of the same color
        self.grade = 100
        # Indicates that it has already reached 5, reaching the maximum weight
        self.max_value = 10012345
        self.init_board_flag = False

    def location_to_action(self, i, j):
        """
        Overview:
            Convert coordinate to serial number.
        Arguments:
            - i (:obj:`Int`): X-axis.
            - j (:obj:`Int`): Y-axis.
        Returns:
            - action (:obj:`Int`): The serial number of the entered coordinates on the chessboard.
        """
        # location = (i,j), action=j+i*width
        return j + i * self.board_width

    def action_to_location(self, action):
        """
        Overview:
            Convert serial number to coordinate.
        Arguments:
            - action (:obj:`Int`): The serial number of the entered coordinates on the chessboard.

        Returns:
            - [i, j].
        """
        # location = (i,j), action=j+i*width
        j = action % self.board_width
        i = action // self.board_width
        return [i, j]

    def get_loc_player(self, i, j):
        """
        Overview:
            Returns the state of the pawn at the given coordinates.
        Arguments:
            - [i, j](:obj:`[Int, Int]`): The coordinate on the chessboard.

        Returns:
            - board_status: \
            0: no pawns, \
            1: player 1, \
            2: player 2.
        """
        action = self.location_to_action(i, j)
        return self.board_status[action]

    def scan_leftright(self, i, j, player):
        """
        Overview:
            Calculate the estimated value of the pawn from left to right when the player moves at (i,j)
        Arguments:
            - i (:obj:`Int`): X-axis.
            - j (:obj:`Int`): Y-axis.
            - player (:obj:`Int`): Current player.

        Returns:
            - value: Situation valuation in this direction.
        """
        # Count the number of consecutive pieces or empty pieces of the current player
        # and get the value in this direction when moving pieces (i, j)
        value = 0
        count = 0
        grade = self.grade
        # scan left
        m, n = i, j - 1
        while n >= 0:
            is_continue, value, grade = self.caculate_once_value(m, n,
                                                                 player,
                                                                 value,
                                                                 grade)
            if is_continue:
                # Continue to move one step to the left
                n = n - 1
            else:
                break
            count += 1
        # Change the direction, the weights are returned to the initial values
        grade = self.grade
        n = j + 1
        while n < self.board_width:
            is_continue, value, grade = self.caculate_once_value(m, n,
                                                                 player,
                                                                 value,
                                                                 grade)
            if is_continue:
                # Continue to move one step to the right
                n = n + 1
            else:
                break
            count += 1
        # Returns the value if there are four consecutive pawn in this direction, otherwise 0
        return value if count >= 4 else 0

    def scan_updown(self, i, j, player):
        """
        Overview:
            Calculate the estimated value of the pawn from up to down when the player moves at (i,j)
        Arguments:
            - i (:obj:`Int`): X-axis.
            - j (:obj:`Int`): Y-axis.
            - player (:obj:`Int`): Current player.

        Returns:
            - value: Situation valuation in this direction.
        """
        value = 0
        count = 0
        # Count the number of consecutive pieces or empty pieces of the current player
        # and get the value in this direction when moving pieces (i, j)
        grade = self.grade

        m, n = i - 1, j
        # scan up
        while m >= 0:
            is_continue, value, grade = self.caculate_once_value(m, n,
                                                                 player,
                                                                 value,
                                                                 grade)
            if is_continue:
                # Continue to move one step to the up
                m = m - 1
            else:
                break
            count += 1
        # Change the direction and change the weight back to the initial value
        grade = self.grade
        m = i + 1
        # scan down
        while m < self.board_height:
            is_continue, value, grade = self.caculate_once_value(m, n,
                                                                 player,
                                                                 value,
                                                                 grade)
            if is_continue:
                # Continue to move one step to the down
                m = m + 1
            else:
                break
            count += 1
        # Returns the value if there are four consecutive pawn in this direction, otherwise 0
        return value if count >= 4 else 0

    def scan_left_updown(self, i, j, player):
        """
        Overview:
            Calculate the estimated value of the pawn from top left to bottom right when the player moves at (i,j)
        Arguments:
            - i (:obj:`Int`): X-axis.
            - j (:obj:`Int`): Y-axis.
            - player (:obj:`Int`): Current player.

        Returns:
            - value: Situation valuation in this direction.
        """
        # Count the number of consecutive pieces or empty pieces of the current player
        # and get the value in this direction when moving pieces (i, j)
        value = 0
        count = 0

        grade = self.grade
        m, n = i - 1, j - 1
        # scan left up
        while m >= 0 and n >= 0:
            is_continue, value, grade = self.caculate_once_value(m, n,
                                                                 player,
                                                                 value,
                                                                 grade)
            if is_continue:
                # Continue to move one step to the left up
                m, n = m - 1, n - 1
            else:
                break
            count += 1

        grade = self.grade
        # right down
        m, n = i + 1, j + 1
        while m < self.board_height and n < self.board_width:
            is_continue, value, grade = self.caculate_once_value(m, n,
                                                                 player,
                                                                 value,
                                                                 grade)
            if is_continue:
                # Continue to move one step down to the right
                m, n = m + 1, n + 1
            else:
                break
            count += 1
        # Returns the value if there are four consecutive pawn in this direction, otherwise 0
        return value if count >= 4 else 0

    def scan_right_updown(self, i, j, player):
        """
        Overview:
            Calculate the estimated value of the pawn from top right to bottom left when the player moves at (i,j)
        Arguments:
            - i (:obj:`Int`): X-axis.
            - j (:obj:`Int`): Y-axis.
            - player (:obj:`Int`): Current player.

        Returns:
            - value: Situation valuation in this direction.
        """
        # Count the number of consecutive pieces or empty pieces of the current player
        # and get the value in this direction when moving pieces (i, j)
        value = 0
        count = 0
        grade = self.grade
        # scan left down
        m, n = i + 1, j - 1
        while m < self.board_height and n >= 0:
            is_continue, value, grade = self.caculate_once_value(m, n,
                                                                 player,
                                                                 value,
                                                                 grade)
            if is_continue:
                m, n = m + 1, n - 1
            else:
                break
            count += 1
        grade = self.grade
        # scan right up
        m, n = i - 1, j + 1
        while m >= 0 and n < self.board_width:
            is_continue, value, grade = self.caculate_once_value(m, n,
                                                                 player,
                                                                 value,
                                                                 grade)
            if is_continue:
                # Continue to move up one step to the right
                m, n = m - 1, n + 1
            else:
                break
            count += 1
        # Returns the value if there are four consecutive pawn in this direction, otherwise 0
        return value if count >= 4 else 0

    def caculate_once_value(self, m, n, player, value, grade):
        """
        Overview:
            Calculate the income brought by the pawns adjacent to the position (m,n) \
            when the player places the pawn at the specified position (i,j) \
            in the current situation
        Arguments:
            - m (:obj:`Int`): x.
            - n (:obj:`Int`): y.
            - player (:obj:`Int`): current chess player.
            - value (:obj:`Int`): The current position (the pawn is at (i,j)) is evaluated.
            - grade (:obj:`Int`): The weight of the pawn in the current position

        Returns:
            - is_continue: Whether there is a pawn of current_player at the current position
            - value: The evaluation value of the move to (i,j)
            - grade: The weight of a single one of our chess pieces
        """
        loc_player = self.get_loc_player(m, n)
        if loc_player == player:
            value += grade
        elif loc_player == 0:
            value += 1
            # When encountering an empty chess, reduce the weight of subsequent black chess
            grade = grade / 10
        else:
            # opponent's pawn
            value -= 5
            # When encountering an opponent's chess, return
            return 0, value, grade
        # 1 means 'is_continue'
        return 1, value, grade

    def evaluate_all_value(self, player):
        """
        Overview:
            Calculate the estimates of all possible positions, \
            and choose the most favorable position from them.
        Arguments:
            - player (:obj:`Int`): current chess player.

        Returns:
            - action: the most favorable action
            - self.action_value[action][4]: Situation valuation under this action
        """
        self.action_value = defaultdict(lambda: [0, 0, 0, 0, 0])
        for action in self.available:
            i, j = self.action_to_location(action)

            self.action_value[action][0] = self.scan_updown(i, j, player)
            self.action_value[action][1] = self.scan_leftright(i, j, player)
            self.action_value[action][2] = self.scan_left_updown(i, j, player)
            self.action_value[action][3] = self.scan_right_updown(i, j, player)

            # Indicates that one direction can already be rushed to 4
            for k in range(4):
                if self.action_value[action][k] >= 390:
                    self.action_value[action][k] = 2000
                elif self.action_value[action][k] >= 302:
                    self.action_value[action][k] = 1000

            # Comprehensive score in all directions
            self.action_value[action][4] = (
                self.action_value[action][0] +
                self.action_value[action][1] +
                self.action_value[action][2] +
                self.action_value[action][3])

        action = max(self.available, key=lambda x: self.action_value[x][4])

        return action, self.action_value[action][4]

    def get_action(self, obs):
        """
        Overview:
            Returns a rule-based expert action.
        Arguments:
            - obs (:obj:`np.array`)

        Returns:
            - expert_action
        """
        self.board = obs

        if self.init_board_flag is False:
            self.board_width = self.board['observation'][0].shape[0]
            self.board_height = self.board['observation'][0].shape[1]
            # the 2 dim indicates which player is the to_play player, 1 means player 1, 2 means player 2
            self.init_board_flag = True
        if self.board['observation'][2][0][0] == 1:
            self.m_player_id = 1
            self.s_player_id = 2
        else:
            self.m_player_id = 2
            self.s_player_id = 1
        # transform observation,action_mask to self.available, self.board_status
        self.available = []
        self.board_status = np.zeros(self.board_width * self.board_height, 'int8')
        for i in range(self.board_width):
            for j in range(self.board_height):
                action = self.location_to_action(i, j)
                if self.board['action_mask'][action] == 1:
                    self.available.append(action)
                if self.board['observation'][0][i][j] == 1:
                    self.board_status[action] = self.m_player_id
                elif self.board['observation'][1][i][j] == 1:
                    self.board_status[action] = self.s_player_id

        m_action, m_value = self.evaluate_all_value(self.m_player_id)

        logging.info("loaction:{loc},value:{value}".format(
                     loc=self.action_to_location(m_action), value=m_value))

        s_action, s_value = self.evaluate_all_value(self.s_player_id)
        logging.info("O_loaction:{loc},value:{value}".format(
                      loc=self.action_to_location(s_action), value=s_value))
        # Block this position if it is better for the opponent to play here
        if m_value >= s_value:
            return m_action
        else:
            return s_action
