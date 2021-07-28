import gym
import gym_gomoku
import numpy as np
import sys

from dizoo.chess_games.base_game_env import BaseGameEnv
from ding.envs import BaseEnv, BaseEnvInfo, BaseEnvTimestep
from ding.envs.common.env_element import EnvElement, EnvElementInfo
from ding.utils import ENV_REGISTRY


@ENV_REGISTRY.register('gomoku')
class GomokuEnv(BaseGameEnv):
    def __init__(self, cfg={}):
        self.cfg = cfg
        self.board_size = self.cfg.get('board_size', 15)
        self.player = 1
        self.board = np.zeros((self.board_size, self.board_size), dtype="int32")
        self.board_markers = [
            chr(x) for x in range(ord("A"), ord("A") + self.board_size)
        ]

    def to_play(self):
        return 0 if self.player == 1 else 1

    def reset(self):
        self.player = 1
        self.board = np.zeros((self.board_size, self.board_size), dtype="int32")
        return {'obs': self.board, 'mask': self.legal_actions()}

    def do_action(self,action):
        row, col = self.action_to_coord(action)
        self.player *= -1
        self.board[row, col] = self.player

    def step(self, action):
        row, col = self.action_to_coord(action)
        self.board[row, col] = self.player
        obs = {'obs': self.board, 'mask': self.legal_actions()}

        done = self.is_finished()
        reward = 1 if done else 0
        self.player *= -1
        info = {'next_player': self.player}
        return BaseEnvTimestep(obs, reward, done, info)

    def legal_actions(self):
        ''' Get all the next legal action, namely empty space that you can place your 'color' stone
             Return: Coordinate of all the empty space, [(x1, y1), (x2, y2), ...]
         '''
        legal_actions = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if (self.board[i][j] == 0):
                    legal_actions.append(self.coord_to_action(i, j))
        return legal_actions

    def legal_moves(self):
        legal_moves = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if (self.board[i][j] == 0):
                    legal_moves.append((i, j))
        return legal_moves

    def coord_to_action(self, i, j):
        ''' convert coordinate i, j to action a in [0, board_size**2)
        '''
        a = i * self.board_size + j  # action index
        return a

    def action_to_coord(self, a):
        coord = (a // self.board_size, a % self.board_size)
        return coord

    def is_finished(self):
        has_legal_actions = False
        directions = ((1, -1), (1, 0), (1, 1), (0, 1))
        for i in range(self.board_size):
            for j in range(self.board_size):
                # if no stone is on the position, don't need to consider this position
                if self.board[i][j] == 0:
                    has_legal_actions = True
                    continue
                # value-value at a coord, i-row, j-col
                player = self.board[i][j]
                # check if there exist 5 in a line
                for d in directions:
                    x, y = i, j
                    count = 0
                    for _ in range(5):
                        if (x not in range(self.board_size)) or (
                                y not in range(self.board_size)
                        ):
                            break
                        if self.board[x][y] != player:
                            break
                        x += d[0]
                        y += d[1]
                        count += 1
                        # if 5 in a line, store positions of all stones, return value
                        if count == 5:
                            return True
        return not has_legal_actions

    def game_end(self):
        end = self.is_finished()
        winner = self.player if end else -1
        return end, winner

    def seed(self, seed: int) -> None:
        pass

    def expert_action(self):
        action_list = self.legal_actions()
        return np.random.choice(action_list)

    def render(self):
        marker = "  "
        for i in range(self.board_size):
            marker = marker + self.board_markers[i] + " "
        print(marker)
        for row in range(self.board_size):
            print(chr(ord("A") + row), end=" ")
            for col in range(self.board_size):
                ch = self.board[row][col]
                if ch == 0:
                    print(".", end=" ")
                elif ch == 1:
                    print("X", end=" ")
                elif ch == -1:
                    print("O", end=" ")
            print()

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.
        Returns:
            An integer from the action space.
        """
        while True:
            try:
                row = int(
                    input(
                        f"Enter the row (from top to bottom) to play for the player {self.to_play()}: "
                    )
                )
                col = int(
                    input(
                        f"Enter the column (from left to right) to play for the player {self.to_play()}: "
                    )
                )
                choice = (row - 1) * self.board_size + (col - 1)
                if (
                        choice in self.legal_actions()
                        and 1 <= row
                        and 1 <= col
                        and row <= self.board_size
                        and col <= self.board_size
                ):
                    break
            except KeyboardInterrupt:
                sys.exit(0)
            except Exception as e:
                print("Wrong input, try again")
        return choice

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.
        Returns:
            String representing the action.
        """
        row = action_number // self.board_size+ 1
        col = action_number % self.board_size + 1
        return f"Play row {row}, column {col}"

    def info(self) -> BaseEnvInfo:
        T = EnvElementInfo
        return BaseEnvInfo(
            agent_num=2,
            obs_space=T(
                (self.board_size, self.board_size, 119),
                {
                    'min': -1,
                    'max': 1,
                    'dtype': np.int32,
                },
            ),
            # [min, max)
            act_space=T(
                (1,),
                {
                    'min': 0,
                    'max': self.board_size ** 2,
                    'dtype': int,
                },
            ),
            rew_space=T(
                (1,),
                {
                    'min': -1.0,
                    'max': 1.0
                },
            ),
            use_wrappers=None,
        )

    def __repr__(self) -> str:
        return 'chess'


if __name__ == '__main__':
    env = GomokuEnv()
    env.reset()
    done = False
    while True:
        env.render()
        action = env.human_to_action()
        obs, reward, done, info = env.step(action)
        if done:
            env.render()
            if reward > 0:
                print('human player win')
            else:
                print('draw')
            break
        env.render()
        action = env.expert_action()
        print('computer player ' + env.action_to_string(action))
        obs, reward, done, info = env.step(action)
        if done:
            if reward > 0:
                print('computer player win')
            else:
                print('draw')
            break
