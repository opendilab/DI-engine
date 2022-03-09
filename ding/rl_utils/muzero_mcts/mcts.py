import random

import numpy as np
import torch

import ding.rl_utils.muzero_mcts.ptree as tree


class MCTS(object):
    def __init__(self, config):
        self.config = config

    def search(self, roots, model, hidden_state_roots, ):
        """Do MCTS for the roots (a batch of root nodes in parallel). Parallel in model inference
        Parameters
        ----------
        roots: Any
            a batch of expanded root nodes
        hidden_state_roots: list
            the hidden states of the roots
        reward_hidden_roots: list
            the value prefix hidden states in LSTM of the roots
        """
        with torch.no_grad():
            model.eval()

            # preparation
            num = roots.num
            device = self.config.device
            pb_c_base, pb_c_init, discount = self.config.pb_c_base, self.config.pb_c_init, self.config.discount
            # the data storage of hidden states: storing the states of all the tree nodes
            hidden_state_pool = [hidden_state_roots]
            # 1 x batch x 64

            # the index of each layer in the tree
            hidden_state_index_x = 0
            # minimax value storage
            min_max_stats_lst = tree.MinMaxStatsList(num)

            for index_simulation in range(self.config.num_simulations):
                hidden_states = []

                # prepare a result wrapper to transport results between python and c++ parts
                results = tree.SearchResults(num=num)
                # traverse to select actions for each root
                # hidden_state_index_x_lst: the first index of leaf node states in hidden_state_pool
                # hidden_state_index_y_lst: the second index of leaf node states in hidden_state_pool
                # the hidden state of the leaf node is hidden_state_pool[x, y]; value prefix states are the same
                hidden_state_index_x_lst, hidden_state_index_y_lst, last_actions = tree.batch_traverse(roots, pb_c_base,
                                                                                                       pb_c_init,
                                                                                                       discount,
                                                                                                       min_max_stats_lst,
                                                                                                       results)

                # obtain the states for leaf nodes
                for ix, iy in zip(hidden_state_index_x_lst, hidden_state_index_y_lst):
                    hidden_states.append(hidden_state_pool[ix][iy])

                hidden_states = torch.from_numpy(np.asarray(hidden_states)).to(device).float()

                last_actions = torch.from_numpy(np.asarray(last_actions)).to(device).unsqueeze(1).long()

                # evaluation for leaf nodes
                network_output = model.recurrent_inference(hidden_states, last_actions)

                hidden_state_nodes = network_output.hidden_state
                reward_pool = network_output.reward.reshape(-1).tolist()
                value_pool = network_output.value.reshape(-1).tolist()
                policy_logits_pool = network_output.policy_logits.tolist()

                hidden_state_pool.append(hidden_state_nodes)
                # reset 0
                # reset the hidden states in LSTM every horizon steps in search
                # only need to predict the value prefix in a range (eg: s0 -> s5)

                hidden_state_index_x += 1

                # backpropagation along the search path to update the attributes
                tree.batch_back_propagate(hidden_state_index_x, discount,
                                          reward_pool, value_pool, policy_logits_pool,
                                          min_max_stats_lst, results,)


if __name__ == '__main__':
    import os
    import yaml
    import EasyDict

    default_config_path = os.path.join(os.path.dirname(__file__), 'mcts_config.yaml')
    with open(default_config_path, "r") as f:
        config = yaml.safe_load(f)

    config = EasyDict(config)
    mcts_cfg = config.MCTS
    batch_size = mcts_cfg.batch_size

    model = torch.nn.Linear(in_features=100, out_features=100)

    policy_logits_pool = [random.random() for _ in range(batch_size)]
    hidden_state_roots = [torch.rand(size=(100)) for _ in range(batch_size)]
    roots = tree.Roots(mcts_cfg.batch_size, mcts_cfg.action_space_size, mcts_cfg.num_simulation)
    noises = [np.random.dirichlet([mcts_cfg.root_dirichlet_alpha] * mcts_cfg.action_space_size).astype(
        np.float32).tolist() for _ in range(batch_size)]
    roots.prepare(mcts_cfg.root_exploration_fraction, noises, policy_logits_pool)

    MCTS(mcts_cfg).search(roots, model, hidden_state_roots, )
    roots_distributions = roots.get_distributions()
