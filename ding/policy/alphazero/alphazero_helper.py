import copy
import math
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn


class Node(object):

    def __init__(self, parent, prior_p: float):
        # Tree Structure
        self._parent = parent
        self._children = {}
        # Search meta data
        self._visit_count = 0
        self._value_sum = 0
        self.prior_p = prior_p

    @property
    def value(self):
        """return current value, used to compute ucb score"""
        if self._visit_count == 0:
            return 0
        return self._value_sum / self._visit_count

    def update(self, value):
        self._visit_count += 1
        self._value_sum += value

    def update_recursive(self, leaf_value):
        if not self.is_root():
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def is_leaf(self):
        """Check if the current node is a leaf node or not."""
        return self._children == {}

    def is_root(self):
        """Check if the current node is a root node or not."""
        return self._parent is None

    @property
    def parent(self):
        return self._parent

    @property
    def children(self):
        return self._children

    @property
    def visit_count(self):
        return self._visit_count


class MCTS(object):

    def __init__(self, cfg):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        n_playout: number of simulations, default to 10000
        """
        self._cfg = cfg

        self._max_moves = self._cfg.get('max_moves', 512)  # for chess and shogi, 722 for Go.
        self._num_simulations = self._cfg.get('num_simulations', 800)

        # UCB formula
        self._pb_c_base = self._cfg.get('pb_c_base', 19652)  # 19652
        self._pb_c_init = self._cfg.get('pb_c_init', 1.25)  # 1.25

        # Root prior exploration noise.
        self._root_dirichlet_alpha = self._cfg.get(
            'root_dirichlet_alpha', 0.3
        )  # 0.3  # for chess, 0.03 for Go and 0.15 for shogi.
        self._root_exploration_fraction = self._cfg.get('root_exploration_fraction', 0.25)  # 0.25

    def get_next_action(self, state, policy_forward_fn, temperature=1.0, sample=True):
        root = Node(None, 1.0)
        self._expand_leaf_node(root, state, policy_forward_fn)
        if sample:
            self._add_exploration_noise(root)
        for n in range(self._num_simulations):
            state_copy = copy.deepcopy(state)
            self._simulate(root, state_copy, policy_forward_fn)

        # calc the move probabilities based on visit counts at the root node
        action_visits = []
        for action in range(state.num_actions):
            if action in root.children:
                action_visits.append((action, root.children[action].visit_count))
            else:
                action_visits.append((action, 0))

        actions, visits = zip(*action_visits)
        action_probs = nn.functional.softmax(1.0 / temperature * np.log(torch.as_tensor(visits) + 1e-10)).numpy()
        # if add_noise:
        #     action = np.random.choice(
        #             actions,
        #             p=0.75*action_probs + 0.25*np.random.dirichlet(0.3*np.ones(len(action_probs)))
        #         )
        # else:
        if sample:
            action = np.random.choice(actions, p=action_probs)
        else:
            action = actions[np.argmax(action_probs)]
        return action, action_probs

    def _simulate(self, node, state, policy_forward_fn):
        """
            Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        while not node.is_leaf():
            action, node = self._select_child(node)
            state.do_action(action)

        end, winner = state.game_end()
        if not end:
            leaf_value = self._expand_leaf_node(node, state, policy_forward_fn)
        else:
            if winner == -1:
                leaf_value = 0
            else:
                leaf_value = 1 if state.current_player == winner else -1

        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)

    # Select the child with the highest UCB score.
    def _select_child(self, node):
        _, action, child = max((self._ucb_score(node, child), action, child) for action, child in node.children.items())
        return action, child

    def _expand_leaf_node(self, node, state, policy_forward_fn):
        action_probs_dict, leaf_value = policy_forward_fn(state)
        for action, prior_p in action_probs_dict.items():
            node.children[action] = Node(parent=node, prior_p=prior_p)
        return leaf_value

    # The score for a node is based on its value, plus an exploration bonus based on
    # the prior.
    def _ucb_score(self, parent: Node, child: Node):
        pb_c = math.log((parent.visit_count + self._pb_c_base + 1) / self._pb_c_base) + self._pb_c_init
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior_p
        value_score = child.value
        return prior_score + value_score

    def _add_exploration_noise(self, node):
        actions = node.children.keys()
        noise = np.random.gamma(self._root_dirichlet_alpha, 1, len(actions))
        frac = self._root_exploration_fraction
        for a, n in zip(actions, noise):
            node.children[a].prior_p = node.children[a].prior_p * (1 - frac) + n * frac
