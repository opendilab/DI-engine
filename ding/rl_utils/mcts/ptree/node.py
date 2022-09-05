"""
The Node and Roots class for MCTS in board games in which we must consider legal_actions and to_play.
"""
import math
import random
from typing import List, Any
import torch

import numpy as np
from scipy.special import softmax


class Node:
    """
     Overview:
         the node base class for mcts.
     Arguments:
     """

    def __init__(self, prior: float, legal_actions: Any = None, action_space_size=9):
        self.prior = prior
        self.legal_actions = legal_actions
        self.action_space_size = action_space_size

        self.is_reset = 0
        self.visit_count = 0
        self.value_sum = 0
        self.best_action = -1
        self.to_play = 0  # default one_player_mode
        self.value_prefix = 0.0
        self.children = {}
        self.children_index = []
        # self.hidden_state_index_x = -1
        # self.hidden_state_index_y = -1
        self.hidden_state_index_x = 0
        self.hidden_state_index_y = 0

    def expand(
            self, to_play: int, hidden_state_index_x: int, hidden_state_index_y: int, value_prefix: float,
            policy_logits: List[float]
    ):
        self.to_play = to_play
        if self.legal_actions is None:
            # TODO
            self.legal_actions = np.arange(len(policy_logits))

        self.hidden_state_index_x = hidden_state_index_x
        self.hidden_state_index_y = hidden_state_index_y
        self.value_prefix = value_prefix

        policy_values = torch.softmax(torch.tensor([policy_logits[a] for a in self.legal_actions]), dim=0).tolist()
        policy = {a: policy_values[i] for i, a in enumerate(self.legal_actions)}
        for action, p in policy.items():
            self.children[action] = Node(p)

    def add_exploration_noise(self, exploration_fraction: float, noises: List[float]):
        """
        Overview:
            add exploration noise to priors
        Arguments:
            - noises (:obj: list): length is len(self.legal_actions)
        """
        for i, a in enumerate(self.legal_actions):
            """
            i in index, a is action, e.g. self.legal_actions = [0,1,2,4,6,8], i=[0,1,2,3,4,5], a=[0,1,2,4,6,8]
            """
            try:
                noise = noises[i]
            except Exception as error:
                print(error)
            child = self.get_child(a)
            prior = child.prior
            child.prior = prior * (1 - exploration_fraction) + noise * exploration_fraction

    def get_mean_q(self, is_root: int, parent_q: float, discount: float):
        """
        Overview:
            get mean q
        Arguments:
            - is_root (:obj:`int`):
        """
        total_unsigned_q = 0.0
        total_visits = 0
        parent_value_prefix = self.value_prefix
        for a in self.legal_actions:
            child = self.get_child(a)
            if child.visit_count > 0:
                true_reward = child.value_prefix - parent_value_prefix
                if self.is_reset == 1:
                    # TODO(pu)
                    true_reward = child.value_prefix
                # TODO(pu): only one step bootstrap?
                q_of_s_a = true_reward + discount * child.value
                total_unsigned_q += q_of_s_a
                total_visits += 1
        if is_root and total_visits > 0:
            mean_q = total_unsigned_q / total_visits
        else:
            # if is not root node,
            # TODO(pu): why parent_q?
            mean_q = (parent_q + total_unsigned_q) / (total_visits + 1)
        return mean_q

    def print_out(self):
        pass

    def get_trajectory(self):
        """
        Overview:
            get best trajectory
        """
        # TODO(pu): best action
        traj = []
        node = self
        best_action = node.best_action
        while best_action >= 0:
            traj.append(best_action)

            node = node.get_child(best_action)
            best_action = node.best_action
        return traj

    def get_children_distribution(self):
        if self.legal_actions == []:
            return None
        distribution = {a: 0 for a in self.legal_actions}
        if self.expanded:
            for a in self.legal_actions:
                child = self.get_child(a)
                distribution[a] = child.visit_count
            # only take the visit counts
            distribution = [v for k, v in distribution.items()]
        return distribution

    def get_child(self, action):
        """
        Overview:
            get children node according to action.
        """
        # assert isinstance(action, int)
        if not isinstance(action, np.int64):
            action = int(action)
        return self.children[action]

    @property
    def expanded(self):
        return len(self.children) > 0

    @property
    def value(self):
        if self.visit_count == 0:
            return 0
        else:
            return self.value_sum / self.visit_count


class Roots:

    def __init__(self, root_num: int, legal_actions_list: Any, pool_size: int):
        self.num = root_num
        self.root_num = root_num
        self.legal_actions_list = legal_actions_list  # list of list
        self.pool_size = pool_size

        self.roots = []
        for i in range(self.root_num):
            if isinstance(legal_actions_list, list):
                self.roots.append(Node(0, legal_actions_list[i]))
            else:
                # if legal_actions_list is int
                self.roots.append(Node(0, np.arange(legal_actions_list)))

    def prepare(self, root_exploration_fraction, noises, value_prefixs, policies, to_play=None):
        for i in range(self.root_num):
            #  to_play: int, hidden_state_index_x: int, hidden_state_index_y: int,
            # TODO(pu): why hidden_state_index_x=0, hidden_state_index_y=i?
            if to_play is None:
                self.roots[i].expand(0, 0, i, value_prefixs[i], policies[i])
            elif to_play is [None]:
                print('debug')
            else:
                self.roots[i].expand(to_play[i], 0, i, value_prefixs[i], policies[i])

            self.roots[i].add_exploration_noise(root_exploration_fraction, noises[i])
            self.roots[i].visit_count += 1

    def prepare_no_noise(self, value_prefixs, policies, to_play=None):
        for i in range(self.root_num):
            if to_play is None:
                self.roots[i].expand(0, 0, i, value_prefixs[i], policies[i])
            else:
                self.roots[i].expand(to_play[i], 0, i, value_prefixs[i], policies[i])

            self.roots[i].visit_count += 1

    def clear(self):
        self.roots.clear()

    def get_trajectories(self):
        trajs = []
        for i in range(self.root_num):
            trajs.append(self.roots[i].get_trajectory())
        return trajs

    def get_distributions(self):
        distributions = []
        for i in range(self.root_num):
            distributions.append(self.roots[i].get_children_distribution())

        return distributions

    def get_values(self):
        values = []
        for i in range(self.root_num):
            values.append(self.roots[i].value)
        return values


class SearchResults:

    def __init__(self, num):
        self.num = num
        self.nodes = []
        self.search_paths = []
        self.hidden_state_index_x_lst = []
        self.hidden_state_index_y_lst = []
        self.last_actions = []
        self.search_lens = []


def update_tree_q(root: Node, min_max_stats, discount: float, players=1):
    node_stack = []
    node_stack.append(root)
    parent_value_prefix = 0.0
    is_reset = 0
    while len(node_stack) > 0:
        node = node_stack[-1]
        node_stack.pop()

        if node != root:
            true_reward = node.value_prefix - parent_value_prefix
            if is_reset == 1:
                true_reward = node.value_prefix
            if players == 1:
                q_of_s_a = true_reward + discount * node.value
            elif players == 2:
                q_of_s_a = true_reward + discount * (-node.value)

            min_max_stats.update(q_of_s_a)
        for a in node.legal_actions:
            child = node.get_child(a)
            if child.expanded:
                node_stack.append(child)
        parent_value_prefix = node.value_prefix
        is_reset = node.is_reset


def back_propagate(search_path, min_max_stats, to_play, value: float, discount: float):
    if to_play is None or to_play == 0:
        # for 1 player mode
        bootstrap_value = value
        path_len = len(search_path)
        for i in range(path_len - 1, -1, -1):
            node = search_path[i]
            node.value_sum += bootstrap_value
            node.visit_count += 1

            parent_value_prefix = 0.0
            is_reset = 0
            if i >= 1:
                parent = search_path[i - 1]
                parent_value_prefix = parent.value_prefix
                is_reset = parent.is_reset

            true_reward = node.value_prefix - parent_value_prefix
            if is_reset == 1:
                true_reward = node.value_prefix

            bootstrap_value = true_reward + discount * bootstrap_value

        min_max_stats.clear()
        root = search_path[0]
        update_tree_q(root, min_max_stats, discount, 1)
    else:
        # for 2 player mode
        bootstrap_value = value
        path_len = len(search_path)
        for i in range(path_len - 1, -1, -1):
            node = search_path[i]
            # to_play related
            node.value_sum += bootstrap_value if node.to_play == to_play else - bootstrap_value

            node.visit_count += 1

            parent_value_prefix = 0.0
            is_reset = 0
            if i >= 1:
                parent = search_path[i - 1]
                parent_value_prefix = parent.value_prefix
                is_reset = parent.is_reset

            true_reward = node.value_prefix - parent_value_prefix
            if is_reset == 1:
                true_reward = node.value_prefix
            # to_play related
            bootstrap_value = (- true_reward if node.to_play == to_play else true_reward) + discount * bootstrap_value

        min_max_stats.clear()
        root = search_path[0]
        update_tree_q(root, min_max_stats, discount, 2)


def batch_back_propagate(
        hidden_state_index_x: int,
        discount: float,
        value_prefixs: List,
        values: List[float],
        policies: List[float],
        min_max_stats_lst,
        results,
        is_reset_lst: List,
        to_play: list = None
) -> None:
    for i in range(results.num):
        #  to_play: int, hidden_state_index_x: int, hidden_state_index_y: int,
        # TODO(pu): why to_play=0, hidden_state_index_x=hidden_state_index_x, hidden_state_index_y=i
        if to_play is None:
            # set to_play=0, because two_player mode to_play = {1,2}
            results.nodes[i].expand(0, hidden_state_index_x, i, value_prefixs[i], policies[i])
        else:
            results.nodes[i].expand(to_play[i], hidden_state_index_x, i, value_prefixs[i], policies[i])
        # reset
        results.nodes[i].is_reset = is_reset_lst[i]
        if to_play is None:
            back_propagate(results.search_paths[i], min_max_stats_lst.stats_lst[i], 0, values[i], discount)
        else:
            back_propagate(results.search_paths[i], min_max_stats_lst.stats_lst[i], to_play[i], values[i], discount)


def select_child(root: Node, min_max_stats, pb_c_base: int, pb_c_int: float, discount: float, mean_q: float,
                 players: int) -> int:
    max_score = -np.inf
    epsilon = 0.000001
    max_index_lst = []
    for a in root.legal_actions:
        child = root.get_child(a)
        temp_score = compute_ucb_score(
            child, min_max_stats, mean_q, root.is_reset, root.visit_count - 1, root.value_prefix, pb_c_base, pb_c_int,
            discount, players
        )
        if max_score < temp_score:
            max_score = temp_score
            max_index_lst.clear()
            max_index_lst.append(a)
        elif temp_score >= max_score - epsilon:
            # TODO(pu): if the difference is less than  epsilon = 0.000001, we random choice action from  max_index_lst
            max_index_lst.append(a)

    action = 0
    if len(max_index_lst) > 0:
        action = random.choice(max_index_lst)
    return action


def compute_ucb_score(
        child: Node,
        min_max_stats,
        parent_mean_q,
        is_reset: int,
        total_children_visit_counts: float,
        parent_value_prefix: float,
        pb_c_base: float,
        pb_c_init: float,
        discount: float,
        players=1
):
    """
    Overview:
        calculate the pUCB score.
    Arguments:
        - child (:obj:`Any`): a child node
        - players (:obj:`int`): one/two_player mode board games
    """
    pb_c = math.log((total_children_visit_counts + pb_c_base + 1) / pb_c_base) + pb_c_init
    pb_c *= (math.sqrt(total_children_visit_counts) / (child.visit_count + 1))

    prior_score = pb_c * child.prior
    if child.visit_count == 0:
        value_score = parent_mean_q
    else:
        true_reward = child.value_prefix - parent_value_prefix
        if is_reset == 1:
            true_reward = child.value_prefix
        if players == 1:
            value_score = true_reward + discount * child.value
        elif players == 2:
            value_score = true_reward + discount * (-child.value)

    value_score = min_max_stats.normalize(value_score)
    if value_score < 0:
        value_score = 0
    if value_score > 1:
        value_score = 1
    ucb_score = prior_score + value_score

    return ucb_score


def batch_traverse(
        roots, pb_c_base: int, pb_c_init: float, discount: float, min_max_stats_lst, results: SearchResults,
        virtual_to_play
):
    last_action = 0
    parent_q = 0.0
    results.search_lens = [None for i in range(results.num)]
    results.last_actions = [None for i in range(results.num)]

    results.nodes = [None for i in range(results.num)]
    results.hidden_state_index_x_lst = [None for i in range(results.num)]
    results.hidden_state_index_y_lst = [None for i in range(results.num)]
    if virtual_to_play is not None and virtual_to_play[0] is not None:
        players = 2
    else:
        players = 1

    results.search_paths = {i: [] for i in range(results.num)}
    for i in range(results.num):
        node = roots.roots[i]
        is_root = 1
        search_len = 0
        results.search_paths[i].append(node)

        # MCTS stage 1:
        # Each simulation starts from the internal root state s0, and finishes when the simulation reaches a leaf node s_l.
        while node.expanded:

            mean_q = node.get_mean_q(is_root, parent_q, discount)
            is_root = 0
            parent_q = mean_q

            # select action according to the pUCT rule
            action = select_child(node, min_max_stats_lst.stats_lst[i], pb_c_base, pb_c_init, discount, mean_q, players)
            if virtual_to_play is not None and virtual_to_play[i] is not None:
                # Players play turn by turn
                if virtual_to_play[i] == 1:
                    virtual_to_play[i] = 2
                else:
                    virtual_to_play[i] = 1
            node.best_action = action
            # move to child node according to action
            node = node.get_child(action)
            last_action = action
            results.search_paths[i].append(node)
            search_len += 1

            # note this return the parent node of the current searched node
            parent = results.search_paths[i][len(results.search_paths[i]) - 1 - 1]

            results.hidden_state_index_x_lst[i] = parent.hidden_state_index_x
            results.hidden_state_index_y_lst[i] = parent.hidden_state_index_y
            results.last_actions[i] = last_action
            results.search_lens[i] = search_len
            results.nodes[i] = node

    # print(f'env {i} one simulation done!')
    return results.hidden_state_index_x_lst, results.hidden_state_index_y_lst, results.last_actions, virtual_to_play
