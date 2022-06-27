"""
The following code is adapted from https://github.com/werner-duvaud/muzero-general
"""

import numpy as np
import torch

import ding.rl_utils.mcts.ptree as tree
from ding.model.template.efficientzero.efficientzero_base_model import inverse_scalar_transform


class EfficientZeroMCTS(object):

    def __init__(self, config):
        self.config = config

    def search(self, roots, model, hidden_state_roots, reward_hidden_roots):
        """
        Overview:
            Do MCTS for the roots (a batch of root nodes in parallel). Parallel in model inference
            Parameters
        Arguments:
            - roots (:obj:`Any`): a batch of expanded root nodes
            - hidden_state_roots (:obj:`list`): the hidden states of the roots
            - reward_hidden_roots(:obj:`list`): the value prefix hidden states in LSTM of the roots
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
            # ez related
            # the data storage of value prefix hidden states in LSTM
            reward_hidden_c_pool = [reward_hidden_roots[0]]
            reward_hidden_h_pool = [reward_hidden_roots[1]]

            # the index of each layer in the tree
            hidden_state_index_x = 0
            # minimax value storage
            min_max_stats_lst = tree.MinMaxStatsList(num)

            horizons = self.config.lstm_horizon_len

            for index_simulation in range(self.config.num_simulations):
                hidden_states = []
                hidden_states_c_reward = []
                hidden_states_h_reward = []

                # prepare a result wrapper to transport results between python and c++ parts
                results = tree.SearchResults(num=num)
                # traverse to select actions for each root
                # hidden_state_index_x_lst: the first index of leaf node states in hidden_state_pool
                # hidden_state_index_y_lst: the second index of leaf node states in hidden_state_pool
                # the hidden state of the leaf node is hidden_state_pool[x, y]; value prefix states are the same
                hidden_state_index_x_lst, hidden_state_index_y_lst, last_actions = tree.batch_traverse(
                    roots, pb_c_base, pb_c_init, discount, min_max_stats_lst, results
                )
                # obtain the search horizon for leaf nodes
                # TODO
                search_lens = results.search_lens

                # obtain the states for leaf nodes
                for ix, iy in zip(hidden_state_index_x_lst, hidden_state_index_y_lst):
                    hidden_states.append(hidden_state_pool[ix][iy])
                    hidden_states_c_reward.append(reward_hidden_c_pool[ix][0][iy])
                    hidden_states_h_reward.append(reward_hidden_h_pool[ix][0][iy])
                    # hidden_states_c_reward.append(reward_hidden_c_pool[ix][iy])
                    # hidden_states_h_reward.append(reward_hidden_h_pool[ix][iy])

                hidden_states = torch.from_numpy(np.asarray(hidden_states)).to(device).float()
                hidden_states_c_reward = torch.from_numpy(np.asarray(hidden_states_c_reward)).to(device).unsqueeze(0)
                hidden_states_h_reward = torch.from_numpy(np.asarray(hidden_states_h_reward)).to(device).unsqueeze(0)

                last_actions = torch.from_numpy(np.asarray(last_actions)).to(device).unsqueeze(1).long()
                try:
                    # evaluation for leaf nodes
                    network_output = model.recurrent_inference(
                        hidden_states, (hidden_states_c_reward, hidden_states_h_reward), last_actions
                    )

                    # TODO(pu)
                    if not model.training:
                        # if not in training, obtain the scalars of the value/reward
                        network_output.value = inverse_scalar_transform(network_output.value, self.config.support_size).detach().cpu().numpy()
                        network_output.value_prefix = inverse_scalar_transform(network_output.value_prefix,
                                                                self.config.support_size).detach().cpu().numpy()
                        network_output.hidden_state = network_output.hidden_state.detach().cpu().numpy()
                        network_output.reward_hidden = (network_output.reward_hidden[0].detach().cpu().numpy(), network_output.reward_hidden[1].detach().cpu().numpy())
                        network_output.policy_logits = network_output.policy_logits.detach().cpu().numpy()

                except Exception as error:
                    print(error)

                hidden_state_nodes = network_output.hidden_state
                value_prefix_pool = network_output.value_prefix.reshape(-1).tolist()
                value_pool = network_output.value.reshape(-1).tolist()
                policy_logits_pool = network_output.policy_logits.tolist()
                reward_hidden_nodes = network_output.reward_hidden

                hidden_state_pool.append(hidden_state_nodes)
                # reset 0
                # reset the hidden states in LSTM every horizon steps in search
                # only need to predict the value prefix in a range (eg: s0 -> s5)
                assert horizons > 0
                reset_idx = (np.array(search_lens) % horizons == 0)
                # assert len(reset_idx) == num
                # if len(reset_idx) != num:
                #     print('here')
                # reset_idx = int(reset_idx.item())
                reward_hidden_nodes[1][:, reset_idx, :] = 0
                is_reset_lst = reset_idx.astype(np.int32).tolist()
                reward_hidden_c_pool.append(reward_hidden_nodes[0])
                reward_hidden_h_pool.append(reward_hidden_nodes[1])
                hidden_state_index_x += 1

                # backpropagation along the search path to update the attributes
                tree.batch_back_propagate(
                    hidden_state_index_x, discount, value_prefix_pool, value_pool, policy_logits_pool,
                    min_max_stats_lst, results, is_reset_lst
                )


class MCTS(object):
    pass
    # TODO
