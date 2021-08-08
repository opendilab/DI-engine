import sys

import numpy as np

from ding.envs.common.env_element import EnvElementInfo
from ding.envs.env.base_env import BaseEnvInfo, BaseEnvTimestep
from ding.utils.registry_factory import ENV_REGISTRY
from dizoo.chess_games.base_game_env import BaseGameEnv


@ENV_REGISTRY.register('tictactoe')
class TicTacToeEnv(BaseGameEnv):
    def __init__(self, cfg=None):
        self.board_height = 3
        self.board_width = 3
        self.players = [1, 2]

    @property
    def current_player(self):
        return self._current_player

    @property
    def current_opponent_player(self):
        return self.players[0] if self.current_player == self.players[1] else self.players[1]

    @property
    def legal_actions(self):
        return self._legal_actions

    def reset(self, start_player=0):
        self.board = np.zeros((self.board_width, self.board_height), dtype="int32")
        self._current_player = self.players[start_player]
        self._legal_actions = list(range(self.board_width * self.board_height))
        return self.current_state()

    def do_action(self, action):
        row = action // 3
        col = action % 3
        self.board[row, col] = self.current_player
        self._current_player = self.current_opponent_player
        self._legal_actions.remove(action)

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
        board_to_play = np.full((3, 3), self.current_player)
        return np.array([board_curr_player, board_opponent_player, board_to_play], dtype="int32")

    def have_winner(self):
        # Horizontal and vertical checks
        for i in range(3):
            if len(set(self.board[i, :])) == 1 and (self.board[i, 0] != 0):
                return True, self.board[i, 0]
            if len(set(self.board[:, i])) == 1 and (self.board[0, i] != 0):
                return True, self.board[0, i]

        # Diagonal checks
        if self.board[0, 0] == self.board[1, 1] == self.board[2, 2] != 0:
            return True, self.board[0, 0]
        if self.board[2, 0] == self.board[1, 1] == self.board[0, 2] != 0:
            return True, self.board[2, 0]

        return False, -1

    def game_end(self):
        """Check whether the game is ended or not"""
        win, winner = self.have_winner()
        if win:
            return True, winner
        elif len(self.legal_actions) == 0:
            return True, -1
        else:
            return False, -1

    def seed(self, seed: int) -> None:
        pass

    def expert_action(self):
        board = self.board
        action = np.random.choice(self.legal_actions)
        # Horizontal and vertical checks
        for i in range(3):
            if abs(sum(board[i, :])) == 2:
                ind = np.where(board[i, :] == 0)[0][0]
                action = np.ravel_multi_index(
                    (np.array([i]), np.array([ind])), (3, 3)
                )[0]
                if self.current_player * sum(board[i, :]) > 0:
                    return action

            if abs(sum(board[:, i])) == 2:
                ind = np.where(board[:, i] == 0)[0][0]
                action = np.ravel_multi_index(
                    (np.array([ind]), np.array([i])), (3, 3)
                )[0]
                if self.current_player * sum(board[:, i]) > 0:
                    return action

        # Diagonal checks
        diag = board.diagonal()
        anti_diag = np.fliplr(board).diagonal()
        if abs(sum(diag)) == 2:
            ind = np.where(diag == 0)[0][0]
            action = np.ravel_multi_index(
                (np.array([ind]), np.array([ind])), (3, 3)
            )[0]
            if self.current_player * sum(diag) > 0:
                return action

        if abs(sum(anti_diag)) == 2:
            ind = np.where(anti_diag == 0)[0][0]
            action = np.ravel_multi_index(
                (np.array([ind]), np.array([2 - ind])), (3, 3)
            )[0]
            if self.current_player * sum(anti_diag) > 0:
                return action

        return action

    def render(self):
        print(self.board[::-1])

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
                        f"Enter the row (1, 2 or 3, from bottom to up) to play for the player {self.current_player}: "
                    )
                )
                col = int(
                    input(
                        f"Enter the column (1, 2 or 3, from left to right) to play for the player {self.current_player}: "
                    )
                )
                choice = (row - 1) * 3 + (col - 1)
                if (
                        choice in self.legal_actions
                        and 1 <= row
                        and 1 <= col
                        and row <= 3
                        and col <= 3
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
        row = action_number // 3 + 1
        col = action_number % 3 + 1
        return f"Play row {row}, column {col}"

    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for mini_batch in play_data:
            state = mini_batch['state']
            mcts_prob = mini_batch['mcts_prob']
            winner = mini_batch['winner']
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_prob.reshape(self.board_height, self.board_width)), i)
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

    def info(self) -> BaseEnvInfo:
        T = EnvElementInfo
        return BaseEnvInfo(
            agent_num=2,
            obs_space={'obs': T(
                (3, 3, 3),
                {
                    'min': -1,
                    'max': 1,
                    'dtype': np.float32,
                },
            ),
                'mask': T(
                    (1,),
                    {
                        'min': 0,
                        'max': 8,
                        'dtype': int,
                    },
                ),
            },
            # [min, max)
            act_space=T(
                (1,),
                {
                    'min': 0,
                    'max': 8,
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

    def close(self) -> None:
        pass

    def __repr__(self) -> str:
        return 'TicTacToe'


if __name__ == '__main__':
    env = TicTacToeEnv()
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
