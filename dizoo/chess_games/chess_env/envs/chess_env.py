import gym
import gym_chess
import numpy as np
import sys

from dizoo.chess_games.base_game_env import BaseGameEnv
from ding.envs import BaseEnv, BaseEnvInfo, BaseEnvTimestep
from ding.envs.common.env_element import EnvElement, EnvElementInfo
from ding.utils import ENV_REGISTRY


@ENV_REGISTRY.register('chess')
class ChessEnv(BaseGameEnv):
    def __init__(self, cfg={}):
        self.cfg = cfg
        self.player = 1
        self.env = gym.make('ChessAlphaZero-v0')

    def to_play(self):
        return 0 if self.player == 1 else 1

    def reset(self):
        self.player = 1
        obs = {'obs': self.env.reset(), 'mask': self.legal_actions()}

        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = {'obs': obs, 'mask': self.legal_actions()}
        return BaseEnvTimestep(obs, reward, done, info)

    def legal_actions(self):
        return self.env.legal_actions

    def legal_moves(self):
        return self.env.legal_moves

    def seed(self, seed: int) -> None:
        pass

    def expert_action(self):
        action_list = self.env.legal_actions
        return np.random.choice(action_list)

    def render(self):
        print(self.env.unwrapped.render())

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.
        Returns:
            An integer from the action space.
        """
        while True:
            try:
                legal_actions = self.legal_actions()
                print(f"Current available actions are:{self.legal_moves()}")
                choice = int(
                    input(
                        f"Enter the index of next move for the player {self.to_play()}: "
                    )
                )
                if choice in range(len(legal_actions)):
                    break
            except KeyboardInterrupt:
                sys.exit(0)
            except Exception as e:
                print("Wrong input, try again")
        return legal_actions[choice]

    def action_to_string(self, action_number):
        """
        Convert an action number to a string representing the action.

        Args:
            action_number: an integer from the action space.
        Returns:
            String representing the action.
        """
        idx = self.legal_actions().count([action_number])
        return f"Play {self.legal_moves()[idx]}"

    def info(self) -> BaseEnvInfo:
        T = EnvElementInfo
        return BaseEnvInfo(
            agent_num=2,
            obs_space={'obs': T(
                (8, 8, 119),
                {
                    'min': 0,
                    'max': 922337203685477580,
                    'dtype': np.int64,
                }, ),
                'mask': T(
                    (1,),
                    {
                        'min': 0,
                        'max': 4672,
                        'dtype': int,
                    },
                ),
            },
            # [min, max)
            act_space=T(
                (1,),
                {
                    'min': 0,
                    'max': 4672,
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
    from dizoo.chess_games.chess_env.envs.chess_env import ChessEnv

    env = ChessEnv()
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
