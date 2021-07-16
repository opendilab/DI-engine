import sys
import numpy as np

from dizoo.chess.base_game_env import BaseGameEnv
from ding.envs import BaseEnv, BaseEnvInfo, BaseEnvTimestep
from ding.envs.common.env_element import EnvElement, EnvElementInfo
from ding.utils import ENV_REGISTRY


@ENV_REGISTRY.register('tictactoe')
class TicTacToeEnv(BaseGameEnv):
    def __init__(self, cfg=None):
        self.board = np.zeros((3, 3), dtype="int32")
        self.player = 1

    def to_play(self):
        return 0 if self.player == 1 else 1

    def reset(self):
        self.board = np.zeros((3, 3), dtype="int32")
        self.player = 1
        obs = {'obs':self.get_observation(),'mask': self.legal_actions()}
        return obs

    def step(self, action):
        row = action // 3
        col = action % 3
        self.board[row, col] = self.player

        done = self.have_winner() or len(self.legal_actions()) == 0

        reward = 1 if self.have_winner() else 0

        self.player *= -1
        info = {'next_player': self.player}
        obs = {'obs':self.get_observation(),'mask': self.legal_actions()}
        return BaseEnvTimestep(obs, reward, done, info)

    def get_observation(self):
        board_player1 = np.where(self.board == 1, 1, 0)
        board_player2 = np.where(self.board == -1, 1, 0)
        board_to_play = np.full((3, 3), self.player)
        return np.array([board_player1, board_player2, board_to_play], dtype="int32")

    def legal_actions(self):
        legal = []
        for i in range(9):
            row = i // 3
            col = i % 3
            if self.board[row, col] == 0:
                legal.append(i)
        return legal

    def have_winner(self):
        # Horizontal and vertical checks
        for i in range(3):
            if (self.board[i, :] == self.player * np.ones(3, dtype="int32")).all():
                return True
            if (self.board[:, i] == self.player * np.ones(3, dtype="int32")).all():
                return True

        # Diagonal checks
        if (
                self.board[0, 0] == self.player
                and self.board[1, 1] == self.player
                and self.board[2, 2] == self.player
        ):
            return True
        if (
                self.board[2, 0] == self.player
                and self.board[1, 1] == self.player
                and self.board[0, 2] == self.player
        ):
            return True

        return False

    def seed(self, seed: int) -> None:
        pass

    def expert_action(self):
        board = self.board
        action = np.random.choice(self.legal_actions())
        # Horizontal and vertical checks
        for i in range(3):
            if abs(sum(board[i, :])) == 2:
                ind = np.where(board[i, :] == 0)[0][0]
                action = np.ravel_multi_index(
                    (np.array([i]), np.array([ind])), (3, 3)
                )[0]
                if self.player * sum(board[i, :]) > 0:
                    return action

            if abs(sum(board[:, i])) == 2:
                ind = np.where(board[:, i] == 0)[0][0]
                action = np.ravel_multi_index(
                    (np.array([ind]), np.array([i])), (3, 3)
                )[0]
                if self.player * sum(board[:, i]) > 0:
                    return action

        # Diagonal checks
        diag = board.diagonal()
        anti_diag = np.fliplr(board).diagonal()
        if abs(sum(diag)) == 2:
            ind = np.where(diag == 0)[0][0]
            action = np.ravel_multi_index(
                (np.array([ind]), np.array([ind])), (3, 3)
            )[0]
            if self.player * sum(diag) > 0:
                return action

        if abs(sum(anti_diag)) == 2:
            ind = np.where(anti_diag == 0)[0][0]
            action = np.ravel_multi_index(
                (np.array([ind]), np.array([2 - ind])), (3, 3)
            )[0]
            if self.player * sum(anti_diag) > 0:
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
                        f"Enter the row (1, 2 or 3, from bottom to up) to play for the player {self.to_play()}: "
                    )
                )
                col = int(
                    input(
                        f"Enter the column (1, 2 or 3, from left to right) to play for the player {self.to_play()}: "
                    )
                )
                choice = (row - 1) * 3 + (col - 1)
                if (
                        choice in self.legal_actions()
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
            'mask':T(
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
