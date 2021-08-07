import numpy as np

from ding.envs import get_env_cls
from ding.policy.model_based.alphazero_helper import MCTS


class AlphazeroCollector:
    def __init__(self, cfg, agent):
        self.cfg = cfg
        self.env_fn = get_env_cls(cfg.env)
        self.env = self.env_fn()
        self.agent = agent
        self.c_puct = self.cfg.collector.c_puct
        self.n_playout = self.cfg.collector.n_playout
        self.mcts = MCTS(self.cfg.mcts)
        self.max_moves = self.cfg.collector.max_moves

    def start_self_play(self, player, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        self.env.reset()
        states, mcts_probs, current_players = [], [], []
        done = False
        while not done and len(states) < self.max_moves:
            move, move_probs = self.mcts.get_next_action(self.env,policy_forward_fn = self.agent.policy_value_function,temperature=1.0)
            # store the data
            states.append(self.env.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.env.current_player)
            # perform a move
            self.env.do_move(move)

            done, winner = self.env.game_end()
            if done:
                # winner from the perspective of the current player of each state
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                player.reset_player()
                if winner != -1:
                    print("Game end. Winner is player:", winner)
                else:
                    print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)

    def get_equi_data(self, play_data):
        return self.env.get_equi_data(play_data)
