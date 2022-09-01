# encoding: utf-8

from collections import defaultdict
import logging
import numpy as np


class Expert(object):
    def __init__(self):
        self.grade = 100
        # 同一色棋子的权重
        self.max_value = 10012345
        # 表示已经已经冲5，达到最大权重
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
        # 统计连续本方旗子或者空棋的个数
        grade = self.grade

        m, n = i, j - 1
        # 向左移动一步
        while n >= 0:
            # 传入落子位置和player
            is_continue, value, grade = self.caculate_once_value(m, n,
                                                                 player,
                                                                 value,
                                                                 grade)
            if is_continue:
                n = n - 1
            else:
                break
                # 向左移动一步
            count += 1

        grade = self.grade
        # 换一个方向，权值调回初始值
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
                # 向右移动一步
            count += 1

        return value if count >= 4 else 0
        # 如果这个方向可成5子，就返回alue，否则就是0

    def scan_updown(self, i, j, player):
        value = 0
        count = 0
        # 统计连续本方旗子或者空棋的个数
        grade = self.grade

        m, n = i - 1, j
        # 向上移动一步
        while m >= 0:
            is_continue, value, grade = self.caculate_once_value(m, n,
                                                                 player,
                                                                 value,
                                                                 grade)
            if is_continue:
                m = m - 1
            else:
                break
                # 向上移动一步
            count += 1

        grade = self.grade
        # 换一个方向，权值调回初始值
        m = i + 1
        # 向下
        while m < self.board_height:
            is_continue, value, grade = self.caculate_once_value(m, n,
                                                                 player,
                                                                 value,
                                                                 grade)
            if is_continue:
                m = m + 1
            else:
                break
                # 向下移动一步
            count += 1

        return value if count >= 4 else 0
        # 如果这个方向可成5子，就返回alue，否则就是0

    def scan_left_updown(self, i, j, player):
        value = 0
        count = 0
        # 统计连续本方旗子或者空棋的个数

        grade = self.grade
        m, n = i - 1, j - 1
        # 向左上移动一步
        while m >= 0 and n >= 0:
            is_continue, value, grade = self.caculate_once_value(m, n,
                                                                 player,
                                                                 value,
                                                                 grade)
            if is_continue:
                m, n = m - 1, n - 1
            else:
                break
                # 向左上继续移动一步
            count += 1

        grade = self.grade
        m, n = i + 1, j + 1
        # 向右下
        while m < self.board_height and n < self.board_width:
            is_continue, value, grade = self.caculate_once_value(m, n,
                                                                 player,
                                                                 value,
                                                                 grade)
            if is_continue:
                m, n = m + 1, n + 1
            else:
                break
                # 向左下继续移动一步
            count += 1

        return value if count >= 4 else 0
        # 如果这个方向可成5子，就返回alue，否则就是0

    def scan_right_updown(self, i, j, player):
        value = 0
        count = 0
        # 统计连续本方旗子或者空棋的个数

        grade = self.grade
        m, n = i + 1, j - 1
        # 向左下移动一步
        while m < self.board_height and n >= 0:
            is_continue, value, grade = self.caculate_once_value(m, n,
                                                                 player,
                                                                 value,
                                                                 grade)
            if is_continue:
                m, n = m + 1, n - 1
            else:
                break
                # 向左下继续移动一步
            count += 1
        grade = self.grade
        m, n = i - 1, j + 1
        # 向右上
        while m >= 0 and n < self.board_width:
            is_continue, value, grade = self.caculate_once_value(m, n,
                                                                 player,
                                                                 value,
                                                                 grade)
            if is_continue:
                m, n = m - 1, n + 1
            else:
                break
                # 向右上继续移动一步
            count += 1

        return value if count >= 4 else 0
        # 如果这个方向可成5子，就返回alue，否则就是0

    def caculate_once_value(self, m, n, player, value, grade):
        # 计算指定位置对该player价值

        # return 0 1 2
        loc_player = self.get_loc_player(m, n)

        if loc_player == player:
            value += grade
        elif loc_player == 0:
            value += 1
            grade = grade / 10
            # 碰到空棋，就降低后续的黑棋权值
        else:
            # 对手的棋子
            value -= 5
            return 0, value, grade
            # 遇到对手棋,结束后续移动
        # 1 means 'is_continue'
        return 1, value, grade

    def evaluate_all_value(self, player):
        # 评估所有空闲位置价值

        self.move_value = defaultdict(lambda: [0, 0, 0, 0, 0])
        for move in self.available:
            i, j = self.move_to_location(move)

            self.move_value[move][0] = self.scan_updown(i, j, player)
            self.move_value[move][1] = self.scan_leftright(i, j, player)
            self.move_value[move][2] = self.scan_left_updown(i, j, player)
            self.move_value[move][3] = self.scan_right_updown(i, j, player)

            # 表示一个方向已经可以是冲5了: 最差分数模式： XOO-OOX :396
            # XOOOO-X :396 ;
#            if (self.move_value[move][0] >= 390 or
#            self.move_value[move][1] >= 390 or
#            self.move_value[move][2] >= 390 or
#            self.move_value[move][3] >= 390 ):
#                return  move, self.max_value

            # 表示一个方向已经可以是冲4了
            for k in range(4):
                if self.move_value[move][k] >= 390:
                    self.move_value[move][k] = 2000
                elif self.move_value[move][k] >= 302:
                    self.move_value[move][k] = 1000

            # 综合各个方向得分
            self.move_value[move][4] = (self.move_value[move][0] +
                                        self.move_value[move][1] +
                                        self.move_value[move][2] +
                                        self.move_value[move][3])

        move = max(self.available, key=lambda x: self.move_value[x][4])

        return move, self.move_value[move][4]

    def get_move(self, obs):
        self.board = obs

        if not self.init_board_flag:
            self.board_width = self.board['observation'][0].shape[0]
            self.board_height = self.board['observation'][0].shape[1]
            # the 2 dim indicates which player is the to_play player, 1 means player 1, 2 means player 2
            if self.board['observation'][2][0][0] == 1:
                self.m_player_id = 1
                self.s_player_id = 2
            else:
                m_player_id = 2
                s_player_id = 1
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
                    self.board_status[move] = m_player_id
                elif self.board['observation'][1][i][j] == 1:
                    self.board_status[move] = s_player_id

        m_move, m_value = self.evaluate_all_value(m_player_id)

        logging.info("loaction:{loc},value:{value}".format(
                     loc=self.move_to_location(m_move), value=m_value))

        s_move, s_value = self.evaluate_all_value(s_player_id)
        logging.info("O_loaction:{loc},value:{value}".format(
                      loc=self.move_to_location(s_move), value=s_value))
        # 如果对手下这里更有利，则堵住
        if m_value >= s_value:
            return m_move
        else:
            return s_move
