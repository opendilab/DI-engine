import os

import numpy as np
from ding.utils.data import default_decollate
from ding.envs import get_env_cls
from ding.policy.model_based.alphazero_helper import MCTS


class AlphazeroCollector:
    def __init__(self, cfg, agent):
        self.cfg = cfg
        self.env_fn = get_env_cls(cfg.env)
        self.env = self.env_fn()
        self.agent = agent
        self.mcts = MCTS(self.cfg.mcts)
        self.max_moves = self.cfg.collector.max_moves

    def start_self_play(self,):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        self.env.reset()
        states, mcts_probs, current_players = [], [], []
        done = False
        while not done and len(states) < self.max_moves:
            action, move_probs = self.mcts.get_next_action( self.env,policy_forward_fn = self.agent.policy_value_fn,temperature=1.0)
            # store the data
            states.append(self.env.current_state())
            mcts_probs.append(move_probs)
            current_players.append(self.env.current_player)
            # perform an action
            self.env.do_action(action)
            done, winner = self.env.game_end()
            if done:
                # winner from the perspective of the current player of each state
                winners = np.zeros(len(current_players),dtype=np.float32)
                if winner != -1:
                    winners[np.array(current_players) == winner] = 1.0
                    winners[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                if winner != -1:
                    print("Game end. Winner is player:", winner)
                else:
                    print("Game end. Tie")
                # mini_batch = {}
                # mini_batch['state'] = states
                # mini_batch['mcts_prob'] = mcts_probs
                # mini_batch['winner'] = winner
                # mini_batch = default_decollate(mini_batch)
                data = [{'state':state,'mcts_prob':mcts_prob,'winner':winner} for state, mcts_prob, winner in zip(states,mcts_probs,winners)]
                # return winner, zip(states,mcts_probs,winners)
                return winner, data
    def get_equi_data(self, play_data):
        return self.env.get_equi_data(play_data)

if __name__ == '__main__':
    from ding.config.config import read_config_yaml
    from ding.policy.model_based.alphazero import AlphaZeroPolicy
    cfg_path = os.path.join(os.getcwd(),'alphazero_config.yaml')
    cfg = read_config_yaml(cfg_path)
    agent = AlphaZeroPolicy(cfg)
    collector = AlphazeroCollector(cfg,agent)
    output = collector.start_self_play()
    print(output)