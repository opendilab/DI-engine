import math
import numpy as np
from typing import List
import copy
from abc import abstractmethod
import torch
import torch.nn as nn

class AbstractChessGame(object):

    def do_move(self,action):
        pass

    @abstractmethod
    def do_action(self,action_id):
        raise NotImplementedError

    @abstractmethod
    def legal_moves(self):
        raise NotImplementedError

    @abstractmethod
    def legal_actions(self):
        """
        Should return the legal actions at each turn, if it is not available, it can return
        the whole action space. At each turn, the game have to be able to handle one of returned actions.

        For complex game where calculating legal moves is too long, the idea is to define the legal actions
        equal to the action space but to return a negative reward if the action is illegal.
        Returns:
            An array of integers, subset of the action space.
        """
        raise NotImplementedError

    @abstractmethod
    def game_end(self):
        raise NotImplementedError

    @abstractmethod
    def get_current_player(self):
        raise NotImplementedError

    @abstractmethod
    def to_play(self):
        raise NotImplementedError


class Node(object):
    def __init__(self, parent, prior_p: float, to_play):
        # Tree Structure
        self.parent = parent
        self.children = {}

        # Search meta data
        self.visit_count = 0
        self.value_sum = 0
        self.prior_p = prior_p

    def value(self):
        """return current value, used to compute ucb score"""
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self,action_priors):
        for action,prob in action_priors:
            if action not in self.children:
                self.children[action] = Node(self, prob)

    def update(self,value):
        self.visit_count += 1
        self.total_value += value

    def update_recursive(self,leaf_value):
        if not self.is_root():
            self.parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def is_leaf(self):
        """Check if the current node is a leaf node or not."""
        return self.children == {}

    def is_root(self):
        """Check if the current node is a root node or not."""
        return self.parent is None



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
        self.cfg = cfg
        self._root = Node(None, 1.0)

        self.num_sampling_moves = 30
        self.max_moves = 512  # for chess and shogi, 722 for Go.
        self.num_simulations = 800

        # UCB formula
        self.pb_c_base = self.cfg.pb_c_base  # 19652
        self.pb_c_init = self.cfg.pb_c_init  # 1.25

        # Root prior exploration noise.
        self.root_dirichlet_alpha = self.cfg.root_dirichlet_alpha # 0.3  # for chess, 0.03 for Go and 0.15 for shogi.
        self.root_exploration_fraction = self.cfg.root_exploration_fraction # 0.25

    def get_next_action(self,state,policy_forward_fn,temperature=1.0):
        for n in range(self.num_simulations):
            state_copy = copy.deepcopy(state)
            self.simulate(state_copy,policy_forward_fn)

        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node.visit_count)
                      for act, node in self._root.children.items()]
        acts, visits = zip(*act_visits)
        act_probs = nn.softmax(1.0/temperature * np.log(torch.as_tensor(visits) + 1e-10))
        return act_probs

    def simulate(self, state,policy_forward_fn):
        """
            Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        while not node.is_leaf():
            action,node = self.select_child(node)
            state.do_action(action)

        end, winner = state.game_end()
        if not end:
            leaf_value = self.expand_leaf_node(node,state,policy_forward_fn)
        else:
            if winner == -1:
                leaf_value = 0
            else:
                leaf_value = 1 if state.get_current_player() == winner else -1

        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)


    # Select the child with the highest UCB score.
    def select_child(self, node):
        _, action, child = max((self.ucb_score(node,child),action,child) for action,child in node.children.items())
        return action, child

    def expand_leaf_node(self, node,state,policy_forward_fn):
        laef_value, policy_probs = policy_forward_fn(state)
        for action, prior_p in policy_probs.items():
            node.children[action] = Node(parent=node,prior_p=prior_p)
        return laef_value

    # The score for a node is based on its value, plus an exploration bonus based on
    # the prior.
    def ucb_score(self, parent: Node, child: Node):
        pb_c = math.log((parent.visit_count + self.pb_c_base + 1) /
                        self.pb_c_base) + self.pb_c_init
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior_p
        value_score = child.value()
        return prior_score + value_score

    def add_exploration_noise(self, node):
        actions = node.children.keys()
        noise = np.random.gamma(self.root_dirichlet_alpha, 1, len(actions))
        frac = self.root_exploration_fraction
        for a, n in zip(actions, noise):
            node.children[a].prior_p = node.children[a].prior_p * (1 - frac) + n * frac

    def get_move_probs(self, state, temp=1e-3):
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        for n in range(self.num_simulations):
            state_copy = copy.deepcopy(state)
            self.simulate(state_copy)

        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node.visit_count)
                      for act, node in self._root.children.items()]
        acts, visits = zip(*act_visits)
        act_probs = nn.softmax(1.0/temp * np.log(torch.as_tensor(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = Node(None, 1.0)