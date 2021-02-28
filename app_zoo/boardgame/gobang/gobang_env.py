from typing import Any, List, Union, Optional
import time
import copy
import numpy as np
from nervex.envs import BaseEnv, register_env, BaseEnvTimestep, BaseEnvInfo
from nervex.envs.common.env_element import EnvElement, EnvElementInfo
from nervex.torch_utils import to_tensor, to_ndarray, to_list


class Board(object):
    r"""
    board for the game
    """

    def __init__(self, cfg):
        self.use_torch = True
        self.width = int(cfg.get('width', 8))
        self.height = int(cfg.get('height', 8))
        # board states stored as a dict,
        # key: move as location on the board,
        # value: player as pieces type
        self.states = {}
        # need how many pieces in a row to win
        self.n_in_row = int(cfg.get('n_in_row', 5))
        self.players = [1, 2]  # player1 and player2

    def init_board(self, start_player=0):
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise Exception('board width and height can not be ' 'less than {}'.format(self.n_in_row))
        self.current_player = self.players[start_player]  # start player
        # keep available moves in a list
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1

    def move_to_location(self, move):
        """
        Example：
            3*3 board's moves like:
            6 7 8
            3 4 5
            0 1 2
            and move 5's location is (1,2)
        """
        h = move // self.width
        w = move % self.width
        return [h, w]

    def location_to_move(self, location):
        if len(location) != 2:
            raise RuntimeError(
                "Wrong dimension location input, location dim should be 2 instead of {}".format(len(location))
            )
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            raise RuntimeError("Wrong location input, location out of bound")
        return move

    def current_state(self):
        """return the board state from the perspective of the current player.
        state shape: 4*width*height
        """

        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(zip(*self.states.items())))
            move_curr = moves[players == self.current_player]
            move_oppo = moves[players != self.current_player]
            square_state[0][move_curr // self.width, move_curr % self.height] = 1.0
            square_state[1][move_oppo // self.width, move_oppo % self.height] = 1.0
            # indicate the last move location
            square_state[2][self.last_move // self.width, self.last_move % self.height] = 1.0
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0  # indicate the colour to play
        ret = square_state[:, ::-1, :]
        return ret

    def do_move(self, move):
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = (self.players[0] if self.current_player == self.players[1] else self.players[1])
        self.last_move = move

    def has_a_winner(self):
        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        moved = list(set(range(width * height)) - set(self.availables))
        if len(moved) < self.n_in_row * 2 - 1:
            return False, -1

        for m in moved:
            h = m // width
            w = m % width
            player = states[m]

            if (w in range(width - n + 1) and len(set(states.get(i, -1) for i in range(m, m + n))) == 1):
                return True, player

            if (h in range(height - n + 1)
                    and len(set(states.get(i, -1) for i in range(m, m + n * width, width))) == 1):
                return True, player

            if (w in range(width - n + 1) and h in range(height - n + 1)
                    and len(set(states.get(i, -1) for i in range(m, m + n * (width + 1), width + 1))) == 1):
                return True, player

            if (w in range(n - 1, width) and h in range(height - n + 1)
                    and len(set(states.get(i, -1) for i in range(m, m + n * (width - 1), width - 1))) == 1):
                return True, player

        return False, -1

    def _forbidden(self, h, w):
        player = self.current_player
        if player == 2:
            return -1
        buffer = ["3" * 11 for _ in range(4)]

        width = self.width
        height = self.height
        states = self.states
        n = self.n_in_row

        restricted = [0, 0]

        # bottom to top
        for i in range(w - n, w + n + 1):
            if i < 0 or i >= width:
                continue
            buffer[0] = buffer[0][:i - w + 5] + str(states.get(i + h * width, 0)) + buffer[0][i - w + 6:]
        buffer[0] = buffer[0][:5] + str(player) + buffer[0][6:]

        for i in range(h - n, h + n + 1):
            if i < 0 or i >= height:
                continue
            buffer[1] = buffer[1][:i - h + 5] + str(states.get(i * width + w, 0)) + buffer[1][i - h + 6:]
        buffer[1] = buffer[1][:5] + str(player) + buffer[1][6:]

        i = h - n - 1
        j = w + n + 1
        for count in range(11):
            i += 1
            j -= 1
            if i < 0 or i >= height or j < 0 or j >= width:
                continue
            buffer[2] = buffer[2][:count] + str(states.get(i * width + j, 0)) + buffer[2][count + 1:]
        buffer[2] = buffer[2][:5] + str(player) + buffer[2][6:]

        i = h - n - 1
        j = w - n - 1
        for count in range(11):
            i += 1
            j += 1
            if i < 0 or i >= height or j < 0 or j >= width:
                continue
            buffer[3] = buffer[3][:count] + str(states.get(i * width + j, 0)) + buffer[3][count + 1:]
        buffer[3] = buffer[3][:5] + str(player) + buffer[3][6:]

        for i in range(4):
            if buffer[i].find("1" * 6) != -1:
                return 1

        for i in range(4):
            before = restricted[1]
            restricted[1] = restricted[1] + 1 if (
                buffer[i].find("011110") != -1 or buffer[i].find("011112") != -1 or buffer[i].find("211110") != -1
                or buffer[i].find("011113") != -1 or buffer[i].find("311110") != -1
            ) else restricted[1]
            restricted[1] = restricted[1] + 1 if (buffer[i].find("10111") != -1) else restricted[1]
            if buffer[i].find("11011") != -1:
                if (not (buffer[i].find("1110110") != -1 or buffer[i].find("0110111") != -1
                         or buffer[i].find("1110111") != -1)):
                    restricted[1] = restricted[1] + 1
                    if (not (buffer[i].find("1110110", buffer[i].find("11011") + 1) != -1
                             or buffer[i].find("0110111", buffer[i].find("11011") + 1) != -1
                             or buffer[i].find("1110111", buffer[i].find("11011") + 1) != -1)):
                        if buffer[i].find("11011", buffer[i].find("11011") + 1) != -1:
                            restricted[1] = restricted[1] + 1
            restricted[1] = restricted[1] + 1 if (buffer[i].find("11101") != -1) else restricted[1]
            if restricted[1] - before > 1:
                restricted[1] = restricted[1] if (
                    buffer[i].find("111010111") != -1 or buffer[i].find("11011011") != -1
                    or buffer[i].find("1011101") != -1
                ) else 1
                if restricted[1] > 1:
                    return 2

        if restricted[1] > 1:
            return 2

        for i in range(4):
            if buffer[i].find("01110") != -1:
                restricted[0] = restricted[0] if (
                    buffer[i].find("100111001") != -1 or buffer[i].find("11101") != -1 or buffer[i].find("10111") != -1
                    or buffer[i].find("10011102") != -1 or buffer[i].find("20111001") != -1
                    or buffer[i].find("2011102") != -1 or buffer[i].find("2011103") != -1
                    or buffer[i].find("3011102") != -1 or buffer[i].find("10011103") != -1
                    or buffer[i].find("30111001") != -1
                ) else restricted[0] + 1
            if buffer[i].find("010110") != -1:
                restricted[0] = restricted[0] if (
                    buffer[i].find("00101101") != -1 or buffer[i].find("10101100") != -1
                    or buffer[i].find("10101101") != -1 or buffer[i].find("10101102") != -1
                    or buffer[i].find("20101101") != -1 or buffer[i].find("10101103") != -1
                    or buffer[i].find("30101101") != -1
                ) else restricted[0] + 1
            if buffer[i].find("011010") != -1:
                restricted[0] = restricted[0] if (
                    buffer[i].find("00110101") != -1 or buffer[i].find("10110100") != -1
                    or buffer[i].find("10110101") != -1 or buffer[i].find("10110102") != -1
                    or buffer[i].find("20110101") != -1 or buffer[i].find("10110103") != -1
                    or buffer[i].find("30110101") != -1
                ) else restricted[0] + 1
        if (restricted[0] > 1):
            return 3

        return 0

    def forbidden(self, m):
        h = m // self.width
        w = m % self.width
        return self._forbidden(h, w)

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        elif not len(self.availables):
            return True, -1
        return False, -1

    def get_current_player(self):
        return self.current_player

    def show(self, player1='Player1', player2='Player2'):
        print(player1, "with X")
        print(player2, "with O")
        print()
        for x in range(self.width):
            print("{0:4}".format(x), end='')
        print('\r\n')
        for i in range(self.height - 1, -1, -1):
            print("{0:2d}".format(i), end='')
            for j in range(self.width):
                loc = i * self.width + j
                p = self.states.get(loc, -1)
                if p == 1:
                    print('X'.center(4), end='')
                elif p == 2:
                    print('O'.center(4), end='')
                else:
                    print('-'.center(4), end='')
            print('\n\r')


class GobangEnv(BaseEnv):

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg

        #configs
        self.width = cfg.get('width', 8)
        self.height = cfg.get('height', 8)
        self.n_in_row = cfg.get('n_in_row', 5)
        self.enable_forbidden = cfg.get('enable_forbidden', True)
        self.enable_mask = cfg.get('enable_mask', True)

        self._is_gameover = False
        self._launch_env_flag = False
        self.over_because_forbidden_move = False

    def _launch_env(self):
        self.board = Board(self._cfg)
        self._launch_env_flag = True

    def reset(self):
        self.close()
        if not self._launch_env_flag:
            self._launch_env()
        self.board.init_board()
        self.over_because_forbidden_move = False
        if self.enable_mask:
            obs = {}
            obs['board'] = self.board.current_state()
            obs['action_mask'] = self.get_action_mask()
        else:
            obs = self.board.current_state()
        return obs

    def close(self):
        self._launch_env_flag = False
        self.board = None

    def seed(self, seed: int) -> None:
        raise NotImplementedError

    def step(self, action: np.ndarray) -> BaseEnvTimestep:
        assert isinstance(action, np.ndarray), type(action)
        action = int(action)
        if self.enable_forbidden:
            if self.over_because_forbidden_move:
                obs_board = self.board.current_state()
                rew = 1.0 if self.board.current_player == 2 else -1.0
                info = {
                    'info': "You Won! Player1 loss due to a forbidden move!"
                } if self.board.current_player == 2 else {
                    'info': "You Lost! Player1 loss due to a forbidden move!"
                }
                done = True
                self.board.current_player = (1 if self.board.current_player == 2 else 2)
                if self.enable_mask:
                    obs = {}
                    obs['board'] = obs_board
                    obs['action_mask'] = self.get_action_mask()
                else:
                    obs = obs_board
                return BaseEnvTimestep(
                    obs,
                    rew,
                    done,
                    info
                )
            if self.board.forbidden(action) > 0:
                obs_board = self.board.current_state()
                rew = -1.0
                done = True
                self.over_because_forbidden_move = True
                self.board.current_player = (1 if self.board.current_player == 2 else 2)
                if self.enable_mask:
                    obs = {}
                    obs['board'] = obs_board
                    obs['action_mask'] = self.get_action_mask()
                else:
                    obs = obs_board
                return BaseEnvTimestep(obs, rew, done, {'loss': "Player1 loss due to a forbidden move"})
        self.board.do_move(action)
        self._is_gameover = self.board.game_end()[0]
        obs_board = self.board.current_state()
        rew = self.get_reward()
        done = self._is_gameover
        if self.enable_mask:
            obs = {}
            obs['board'] = obs_board
            obs['action_mask'] = self.get_action_mask()
        else:
            obs = obs_board
        return BaseEnvTimestep(obs, rew, done, {})

    def get_reward(self):
        if not self.board.has_a_winner()[0]:
            return 0.0
        else:
            if self.board.has_a_winner()[1] == self.board.current_player:
                return -1.0
            elif self.board.has_a_winner()[1] != self.board.current_player:
                return 1.0
            else:
                raise Exception("error in player")

    def get_action_mask(self):
        mask = np.zeros(self.width * self.height)
        mask[self.board.availables] = 1
        return mask

    def get_available(self):
        return copy.deepcopy(self.board.availables)

    def get_board(self):
        return copy.deepcopy(self.board)

    def get_current_state(self):
        return copy.deepcopy(self.board.current_state())

    def get_current_player(self):
        return copy.deepcopy(self.board.get_current_player())

    def get_width(self):
        return copy.deepcopy(self.width)

    def get_height(self):
        return copy.deepcopy(self.height)

    def get_game_end(self):
        return copy.deepcopy(self.board.game_end())

    def info(self) -> BaseEnvInfo:
        T = EnvElementInfo
        return BaseEnvInfo(
            agent_num=1,
            obs_space=T((4, self.width, self.height), {
                'min': 0.0,
                'max': 1.0,
                'dtype': float,
            }, None, None),
            act_space=T((self.height * self.width, ), {
                'min': 0,
                'max': 1
            }, None, None),
            rew_space=T((1, ), {
                'min': -1.0,
                'max': 1.0
            }, None, None),
        )

    def __repr__(self) -> str:
        return "nerveX Gobang Env"


register_env('gobang', GobangEnv)
