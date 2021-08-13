import gym
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
        self.players = [1, 2]
        self.board = np.zeros((self.board_size, self.board_size), dtype="int32")
        self.board_markers = [
            chr(x) for x in range(ord("A"), ord("A") + self.board_size)
        ]
        self.num_actions = self.board_size * self.board_size


    @property
    def current_player(self):
        return self._current_player

    @property
    def current_opponent_player(self):
        return self.players[0] if self.current_player == self.players[1] else self.players[1]


    @property
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

    def reset(self, start_player=0):
        self._current_player = self.players[start_player]
        self.board = np.zeros((self.board_size, self.board_size), dtype="int32")
        return self.current_state()

    def do_action(self,action):
        row, col = self.action_to_coord(action)
        self.board[row, col] = self.current_player
        self._current_player = self.current_opponent_player

    def step(self, action):
        curr_player = self.current_player
        next_player = self.current_opponent_player
        self.do_action(action)

        done, winner = self.game_end()
        reward = int((winner == curr_player))

        info = {'next_player': next_player}
        return BaseEnvTimestep(self.current_state(), reward, done, info)

    def current_state(self):
        board_curr_player = np.where(self.board == self.current_player, 1, 0)
        board_opponent_player = np.where(self.board == self.current_opponent_player, 1, 0)
        board_to_play = np.full((self.board_size, self.board_size), self.current_player)

        return np.array([board_curr_player, board_opponent_player, board_to_play,board_to_play], dtype=np.float32)

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

    def have_winner(self):
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
                            return True, player
        return not has_legal_actions, -1

    def game_end(self):
        end, winner = self.have_winner()
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

    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for data in play_data:
            state = data['state']
            mcts_prob = data['mcts_prob']
            winner = data['winner']
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_prob.reshape(self.board_size, self.board_size)), i)
                extend_data.append({'state': equi_state,
                                    'mcts_prob': np.flipud(equi_mcts_prob).flatten(),
                                    'winner': winner})
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append({'state': equi_state,
                                    'mcts_prob': np.flipud(equi_mcts_prob).flatten(),
                                    'winner': winner})
        return extend_data
    def close(self) -> None:
        pass
    def __repr__(self) -> str:
        return 'gomoku'


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
