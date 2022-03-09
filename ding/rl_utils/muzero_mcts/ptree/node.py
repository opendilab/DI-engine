import math
import random
from typing import List

import numpy as np
from scipy.special import softmax


class Node:
    def __init__(self, prior: float, action_num: int, ):
        self.prior = prior
        self.action_num = action_num

        # self.is_reset = 0
        self.visit_count = 0
        self.value_sum = 0
        self.best_action = -1
        self.to_play = 0
        self.children = {}
        self.hidden_state_index_x = -1
        self.hidden_state_index_y = -1
        self.reward = 0

    def expand(self, to_play: int, hidden_state_index_x: int, hidden_state_index_y: int, reward: float,
               policy_logits: List[float], ):
        self.to_play = to_play
        self.hidden_state_index_x = hidden_state_index_x
        self.hidden_state_index_y = hidden_state_index_y
        self.reward = reward

        priors = softmax(np.array(policy_logits))
        for a in range(self.action_num):
            self.children[a] = Node(prior=priors[a], action_num=self.action_num, )

    def add_exploration_noise(self, exploration_fraction: float, noises: List[float]):
        for a in range(self.action_num):
            noise = noises[a]
            child = self.get_child(action=a)
            prior = child.prior
            child.prior = prior * (1 - exploration_fraction) + noise * exploration_fraction

    def print_out(self):
        pass

    def get_trajectory(self):
        traj = []
        node = self
        best_action = node.best_action
        while best_action >= 0:
            traj.append(best_action)
            node = node.get_child(best_action)
            best_action = node.best_action
        return traj

    def get_children_distribution(self):
        distribution = [0 for _ in range(self.action_num)]
        if self.expanded():
            for a in range(self.action_num):
                child = self.get_child(a)
                distribution[a] = child.visit_count
        return distribution

    def get_child(self, action):
        child = self.children[action]
        return child

    def expanded(self):
        child_num = len(self.children_index)
        return child_num > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        else:
            return self.value_sum / self.visit_count


class Roots:
    def __init__(self, root_num: int, action_num: int, pool_size: int):
        self.root_num = root_num
        self.action_num = action_num
        self.pool_size = pool_size

        self.roots = []
        self.node_pools = []
        for i in range(self.root_num):
            self.node_pools.append([])
            self.roots.append(Node(0, action_num, self.node_pools[i]))

    def prepare(self, root_exploration_fraction, noises, value_prefixs, policies):
        for i in range(self.root_num):
            self.roots[i].expand(0, 0, i, value_prefixs[i], policies[i])
            self.roots[i].add_exploration_noise(root_exploration_fraction, noises[i])
            self.roots[i].visit_count += 1

    def prepare_no_noise(self, value_prefixs, policies):
        for i in range(self.root_num):
            self.roots[i].expand(0, 0, i, value_prefixs[i], policies[i])
            self.roots[i].visit_count += 1

    def clear(self):
        self.node_pools.clear()
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
            values.append(self.roots[i].value())
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
        self.search_paths = []


def update_tree_q(root: Node, min_max_stats, discount: float):
    node_stack = []
    node_stack.append(root)
    parent_value_prefix = 0.0
    is_reset = 0
    while node_stack.size() > 0:
        node = node_stack[-1]
        node_stack.pop()

        if node != root:
            true_reward = node.value_prefix - parent_value_prefix
            if is_reset == 1:
                true_reward = node.value_prefix
            qsa = true_reward + discount * node.value()
            min_max_stats.update(qsa)
        for a in range(node.action_num):
            child = node.get_child(a)
            if child.extended:
                node_stack.append(child)
        parent_value_prefix = node.value_prefix
        is_reset = node.is_reset


# TODO(only consider single player)
def back_propagate(search_path, min_max_stats, to_play: int, value: float, discount: float):
    bootstrap_value = value
    for node in reversed(search_path):
        node.value_sum += bootstrap_value
        node.visit_count += 1
        min_max_stats.update(node.reward + discount * node.value())
        bootstrap_value = node.reward + discount * bootstrap_value


def batch_back_propagate(hidden_state_index_x: int, discount: float,
                         rewards: List, values: List[float],
                         policies: List[float], min_max_stats_lst: List,
                         results: List, ) -> None:
    for i in range(results.num):
        results.nodes[i].expand(0, hidden_state_index_x, i, rewards[i], policies[i])
        back_propagate(results.search_paths[i], min_max_stats_lst.stats_lst[i], 0, values[i], discount)


def select_child(root: Node, min_max_stats, pb_c_base: int, pb_c_int: float, discount: float, ) -> int:
    max_score = -str('inf')
    epsilon = 0.000001
    max_index_lst = []
    for a in range(root.action_num):
        child = root.get_child(a)
        temp_score = ucb_score(child, min_max_stats, root.visit_count - 1,
                               pb_c_base, pb_c_int, discount)
        if max_score < temp_score:
            max_score = temp_score
            max_index_lst.clear()
            max_index_lst.append(a)
        elif temp_score >= max_score - epsilon:
            max_index_lst.append(a)
    action = 0
    if len(max_index_lst) > 0:
        action = random.choice(max_index_lst)
    return action


def ucb_score(child: Node, min_max_stats, total_children_visit_counts: float, pb_c_base: float, pb_c_init: float,
              discount: float):
    pb_c = math.log((total_children_visit_counts + pb_c_base + 1) / pb_c_base) + pb_c_init
    pb_c *= (math.sqrt(total_children_visit_counts) / (child.visit_count + 1))

    prior_score = pb_c * child.prior
    if child.visit_count == 0:
        value_score = min_max_stats.normalize(
            child.reward + discount * child.value())  # TODO(not consider case when we have 2 players)
    else:
        value_score = 0
    ucb_score = prior_score + value_score
    return ucb_score


def batch_traverse(roots, pb_c_base: int, pb_c_init: float, discount: float, min_max_stats_lst, results: SearchResults):
    # set seed
    # timeval
    # t1;
    # gettimeofday( & t1, NULL);
    # srand(t1.tv_usec);

    for i in range(results.num):
        node = roots.roots[i]
        search_len = 0
        results.search_paths[i].append(node)

        while node.expanded():
            action = select_child(node, min_max_stats_lst.stats_lst[i], pb_c_base, pb_c_init, discount, )
            node.best_action = action
            ##  next
            node = node.get_child(action)
            last_action = action
            results.search_paths[i].append(node)
            search_len += 1

            parent = results.search_paths[i][len(results.search_paths[i]) - 2]  # TODO (zsh) why -2

            results.hidden_state_index_x_lst.append(parent.hidden_state_index_x)
            results.hidden_state_index_y_lst.append(parent.hidden_state_index_y)

            results.last_actions.append(last_action)
            results.search_lens.append(search_len)
            results.nodes.append(node)
    return results.cresults.hidden_state_index_x_lst, results.cresults.hidden_state_index_y_lst, results.cresults.last_actions
