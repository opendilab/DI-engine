from typing import Tuple, Callable, Any, List
import math
import numpy as np
import torch
from random import choice


class State(object):
    def legal_actions(self):
        pass

    def is_terminal(self):
        return len(self.legal_actions()) == 0

    def next_state(self, action):
        pass


class Node(object):

    def __init__(self, action, parent, state) -> None:
        # Tree data
        self.state = state
        self.action = action

        # Structure
        self.parent = parent
        self.children = {}
        self.untriedActions = set(self.state.legal_actions())

        # Search meta data
        self.visit_count = 0
        self.value = 0.

    @property
    def fully_expanded(self) -> bool:
        return len(self.untriedActions) == 0

    def expand(self, untried_action: int) -> None:
        state = self.state.next_state(untried_action)
        child_node = Node(action=untried_action, parent=self, state=state)
        self.children[untried_action] = child_node
        self.untriedActions.remove(untried_action)
        return child_node

    def update(self):
        pass

class MCTS(object):
    def __init__(self, tree_policy, default_policy, backup) -> None:
        self.tree_policy = tree_policy
        self.default_policy = default_policy
        self.back_up = backup

    def search(self, root, max_simulation_times=500):
        if root.parent is not None:
            raise ValueError("Root's parent must be None.")

        for _ in range(max_simulation_times):
            node = self.get_next_node(root, self.tree_policy)
            node.reward = self.default_policy(node)
            self.back_up(node)

        return rand_max(root.children.values(), key=lambda x: x.q).action

    @classmethod
    def expand(cls, node):
        action = choice(node.untried_actions)
        return node.expand(action)

    @classmethod
    def best_child(cls, node, tree_policy):
        max_v = -np.inf
        max_l = []
        for item in node.children.values():
            value = tree_policy(item)
            if value == max_v:
                max_l.append(item)
            elif value > max_v:
                max_l = [item]
                max_v = value
        best_action_node = choice(max_l)
        return best_action_node.sample_state()

    @classmethod
    def get_next_node(cls, node, tree_policy):
        while not node.state.is_terminal():
            if node.untried_actions:
                return cls.expand(node)
            else:
                node = cls.best_child(node, tree_policy)
        return node


class UCB1(object):
    """
    The typical bandit upper confidence bounds algorithm.
    """

    def __init__(self, c):
        self.c = c

    def __call__(self, action_node):
        if self.c == 0:  # assert that no nan values are returned
            # for action_node.n = 0
            return action_node.q

        return (action_node.q +
                self.c * np.sqrt(2 * np.log(action_node.parent.n) /
                                 action_node.n))


# class Bellman(object):
#     """
#     A dynamical programming update which resembles the Bellman equation
#     of value iteration.
#     See Feldman and Domshlak (2014) for reference.
#     """
#
#     def __init__(self, gamma):
#         self.gamma = gamma
#
#     def __call__(self, node):
#         """
#         :param node: The node to start the backups from
#         """
#         while node is not None:
#             node.n += 1
#             if isinstance(node, StateNode):
#                 node.q = max([x.q for x in node.children.values()])
#             elif isinstance(node, ActionNode):
#                 n = sum([x.n for x in node.children.values()])
#                 node.q = sum([(self.gamma * x.q + x.reward) * x.n
#                               for x in node.children.values()]) / n
#             node = node.parent


def monte_carlo(node):
    """
    A monte carlo update as in classical UCT.
    See feldman amd Domshlak (2014) for reference.
    :param node: The node to start the backup from
    """
    r = node.reward
    while node is not None:
        node.n += 1
        node.q = ((node.n - 1) / node.n) * node.q + 1 / node.n * r
        node = node.parent

import random
import numpy as np


def rand_max(iterable, key=None):
    """
    A max function that tie breaks randomly instead of first-wins as in
    built-in max().
    :param iterable: The container to take the max from
    :param key: A function to compute tha max from. E.g.:
      >>> rand_max([-2, 1], key=lambda x:x**2
      -2
      If key is None the identity is used.
    :return: The entry of the iterable which has the maximum value. Tie
    breaks are random.
    """
    if key is None:
        key = lambda x: x

    max_v = -np.inf
    max_l = []

    for item, value in zip(iterable, [key(i) for i in iterable]):
        if value == max_v:
            max_l.append(item)
        elif value > max_v:
            max_l = [item]
            max_v = value

    return random.choice(max_l)