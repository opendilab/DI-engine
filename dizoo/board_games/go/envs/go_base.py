# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Code from: https://github.com/tensorflow/minigo

"""
A board is a NxN numpy array.
A Coordinate is a tuple index into the board.
A Move is a (Coordinate c | None).
A PlayerMove is a (Color, Move) tuple

(0, 0) is considered to be the upper left corner of the board, and (18, 0) is the lower left.
"""
import copy
import itertools
import os
from collections import namedtuple

import numpy as np

# from . import coords
from dizoo.board_games.go.envs import coords

N = int(os.environ.get('BOARD_SIZE', 19))

# Represent a board as a numpy array, with 0 empty, 1 is black, -1 is white.
# This means that swapping colors is as simple as multiplying array by -1.
WHITE, EMPTY, BLACK, FILL, KO, UNKNOWN = range(-1, 5)

# Represents "group not found" in the LibertyTracker object
MISSING_GROUP_ID = -1

ALL_COORDS = [(i, j) for i in range(N) for j in range(N)]
EMPTY_BOARD = np.zeros([N, N], dtype=np.int8)


def _check_bounds(c):
    return 0 <= c[0] < N and 0 <= c[1] < N


NEIGHBORS = {(x, y): list(filter(_check_bounds, [
    (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])) for x, y in ALL_COORDS}
DIAGONALS = {(x, y): list(filter(_check_bounds, [
    (x + 1, y + 1), (x + 1, y - 1), (x - 1, y + 1), (x - 1, y - 1)])) for x, y in ALL_COORDS}


class IllegalMove(Exception):
    pass


class PlayerMove(namedtuple('PlayerMove', ['color', 'move'])):
    pass


class PositionWithContext(namedtuple('SgfPosition', ['position', 'next_move', 'result'])):
    pass


def place_stones(board, color, stones):
    for s in stones:
        board[s] = color


def replay_position(position, result):
    """
    Wrapper for a go.Position which replays its history.
    Assumes an empty start position! (i.e. no handicap, and history must be exhaustive.)

    Result must be passed in, since a resign cannot be inferred from position
    history alone.

    for position_w_context in replay_position(position):
        print(position_w_context.position)
    """
    assert position.n == len(position.recent), "Position history is incomplete"
    pos = Position(komi=position.komi)
    for player_move in position.recent:
        color, next_move = player_move
        yield PositionWithContext(pos, next_move, result)
        pos = pos.play_move(next_move, color=color)


def find_reached(board, c):
    color = board[c]
    chain = {c}
    reached = set()
    frontier = [c]
    while frontier:
        current = frontier.pop()
        chain.add(current)
        for n in NEIGHBORS[current]:
            if board[n] == color and n not in chain:
                frontier.append(n)
            elif board[n] != color:
                reached.add(n)
    return chain, reached


def is_koish(board, c):
    'Check if c is surrounded on all sides by 1 color, and return that color'
    if board[c] != EMPTY:
        return None
    neighbors = {board[n] for n in NEIGHBORS[c]}
    if len(neighbors) == 1 and EMPTY not in neighbors:
        return list(neighbors)[0]
    else:
        return None


def is_eyeish(board, c):
    'Check if c is an eye, for the purpose of restricting MC rollouts.'
    # pass is fine.
    if c is None:
        return
    color = is_koish(board, c)
    if color is None:
        return None
    diagonal_faults = 0
    diagonals = DIAGONALS[c]
    if len(diagonals) < 4:
        diagonal_faults += 1
    for d in diagonals:
        if not board[d] in (color, EMPTY):
            diagonal_faults += 1
    if diagonal_faults > 1:
        return None
    else:
        return color


class Group(namedtuple('Group', ['id', 'stones', 'liberties', 'color'])):
    """
    stones: a frozenset of Coordinates belonging to this group
    liberties: a frozenset of Coordinates that are empty and adjacent to this group.
    color: color of this group
    """

    def __eq__(self, other):
        return self.stones == other.stones and self.liberties == other.liberties and self.color == other.color


class LibertyTracker():
    @staticmethod
    def from_board(board):
        board = np.copy(board)
        curr_group_id = 0
        lib_tracker = LibertyTracker()
        for color in (WHITE, BLACK):
            while color in board:
                curr_group_id += 1
                found_color = np.where(board == color)
                coord = found_color[0][0], found_color[1][0]
                chain, reached = find_reached(board, coord)
                liberties = frozenset(r for r in reached if board[r] == EMPTY)
                new_group = Group(curr_group_id, frozenset(
                    chain), liberties, color)
                lib_tracker.groups[curr_group_id] = new_group
                for s in chain:
                    lib_tracker.group_index[s] = curr_group_id
                place_stones(board, FILL, chain)

        lib_tracker.max_group_id = curr_group_id

        liberty_counts = np.zeros([N, N], dtype=np.uint8)
        for group in lib_tracker.groups.values():
            num_libs = len(group.liberties)
            for s in group.stones:
                liberty_counts[s] = num_libs
        lib_tracker.liberty_cache = liberty_counts

        return lib_tracker

    def __init__(self, group_index=None, groups=None, liberty_cache=None, max_group_id=1):
        # group_index: a NxN numpy array of group_ids. -1 means no group
        # groups: a dict of group_id to groups
        # liberty_cache: a NxN numpy array of liberty counts
        self.group_index = group_index if group_index is not None else - \
            np.ones([N, N], dtype=np.int32)
        self.groups = groups or {}
        self.liberty_cache = liberty_cache if liberty_cache is not None else np.zeros([
                                                                                      N, N], dtype=np.uint8)
        self.max_group_id = max_group_id

    def __deepcopy__(self, memodict={}):
        new_group_index = np.copy(self.group_index)
        new_lib_cache = np.copy(self.liberty_cache)
        # shallow copy
        new_groups = copy.copy(self.groups)
        return LibertyTracker(new_group_index, new_groups, liberty_cache=new_lib_cache, max_group_id=self.max_group_id)

    def add_stone(self, color, c):
        assert self.group_index[c] == MISSING_GROUP_ID
        captured_stones = set()
        opponent_neighboring_group_ids = set()
        friendly_neighboring_group_ids = set()
        empty_neighbors = set()

        for n in NEIGHBORS[c]:
            neighbor_group_id = self.group_index[n]
            if neighbor_group_id != MISSING_GROUP_ID:
                neighbor_group = self.groups[neighbor_group_id]
                if neighbor_group.color == color:
                    friendly_neighboring_group_ids.add(neighbor_group_id)
                else:
                    opponent_neighboring_group_ids.add(neighbor_group_id)
            else:
                empty_neighbors.add(n)

        new_group = self._merge_from_played(
            color, c, empty_neighbors, friendly_neighboring_group_ids)

        # new_group becomes stale as _update_liberties and
        # _handle_captures are called; must refetch with self.groups[new_group.id]
        for group_id in opponent_neighboring_group_ids:
            neighbor_group = self.groups[group_id]
            if len(neighbor_group.liberties) == 1:
                captured = self._capture_group(group_id)
                captured_stones.update(captured)
            else:
                self._update_liberties(group_id, remove={c})

        self._handle_captures(captured_stones)

        # suicide is illegal
        if len(self.groups[new_group.id].liberties) == 0:
            raise IllegalMove(f"Move at {c} would commit suicide!\n")

        return captured_stones

    def _merge_from_played(self, color, played, libs, other_group_ids):
        stones = {played}
        liberties = set(libs)
        for group_id in other_group_ids:
            other = self.groups.pop(group_id)
            stones.update(other.stones)
            liberties.update(other.liberties)

        if other_group_ids:
            liberties.remove(played)
        assert stones.isdisjoint(liberties)
        self.max_group_id += 1
        result = Group(
            self.max_group_id,
            frozenset(stones),
            frozenset(liberties),
            color)
        self.groups[result.id] = result

        for s in result.stones:
            self.group_index[s] = result.id
            self.liberty_cache[s] = len(result.liberties)

        return result

    def _capture_group(self, group_id):
        dead_group = self.groups.pop(group_id)
        for s in dead_group.stones:
            self.group_index[s] = MISSING_GROUP_ID
            self.liberty_cache[s] = 0
        return dead_group.stones

    def _update_liberties(self, group_id, add=set(), remove=set()):
        group = self.groups[group_id]
        new_libs = (group.liberties | add) - remove
        self.groups[group_id] = Group(
            group_id, group.stones, new_libs, group.color)

        new_lib_count = len(new_libs)
        for s in self.groups[group_id].stones:
            self.liberty_cache[s] = new_lib_count

    def _handle_captures(self, captured_stones):
        for s in captured_stones:
            for n in NEIGHBORS[s]:
                group_id = self.group_index[n]
                if group_id != MISSING_GROUP_ID:
                    self._update_liberties(group_id, add={s})


class Position():
    def __init__(self, board=None, n=0, komi=7.5, caps=(0, 0),
                 lib_tracker=None, ko=None, recent=tuple(),
                 board_deltas=None, to_play=BLACK):
        """
        board: a numpy array
        n: an int representing moves played so far
        komi: a float, representing points given to the second player.
        caps: a (int, int) tuple of captures for B, W.
        lib_tracker: a LibertyTracker object
        ko: a Move
        recent: a tuple of PlayerMoves, such that recent[-1] is the last move.
        board_deltas: a np.array of shape (n, go.N, go.N) representing changes
            made to the board at each move (played move and captures).
            Should satisfy next_pos.board - next_pos.board_deltas[0] == pos.board
        to_play: BLACK or WHITE
        """
        assert type(recent) is tuple
        self.board = board if board is not None else np.copy(EMPTY_BOARD)
        # With a full history, self.n == len(self.recent) == num moves played
        self.n = n
        self.komi = komi
        self.caps = caps
        self.lib_tracker = lib_tracker or LibertyTracker.from_board(self.board)
        self.ko = ko
        self.recent = recent
        self.board_deltas = board_deltas if board_deltas is not None else np.zeros([
                                                                                   0, N, N], dtype=np.int8)
        self.to_play = to_play

    def __deepcopy__(self, memodict={}):
        new_board = np.copy(self.board)
        new_lib_tracker = copy.deepcopy(self.lib_tracker)
        return Position(new_board, self.n, self.komi, self.caps, new_lib_tracker, self.ko, self.recent, self.board_deltas, self.to_play)

    def __str__(self, colors=True):
        if colors:
            pretty_print_map = {
                WHITE: '\x1b[0;31;47mO',
                EMPTY: '\x1b[0;31;43m.',
                BLACK: '\x1b[0;31;40mX',
                FILL: '#',
                KO: '*',
            }
        else:
            pretty_print_map = {
                WHITE: 'O',
                EMPTY: '.',
                BLACK: 'X',
                FILL: '#',
                KO: '*',
            }
        board = np.copy(self.board)
        captures = self.caps
        if self.ko is not None:
            place_stones(board, KO, [self.ko])
        raw_board_contents = []
        for i in range(N):
            row = [' ']
            for j in range(N):
                appended = '<' if (self.recent and (i, j)
                                   == self.recent[-1].move) else ' '
                row.append(pretty_print_map[board[i, j]] + appended)
                if colors:
                    row.append('\x1b[0m')

            raw_board_contents.append(''.join(row))

        row_labels = ['%2d' % i for i in range(N, 0, -1)]
        annotated_board_contents = [''.join(r) for r in zip(
            row_labels, raw_board_contents, row_labels)]
        header_footer_rows = [
            '   ' + ' '.join('ABCDEFGHJKLMNOPQRST'[:N]) + '   ']
        annotated_board = '\n'.join(itertools.chain(
            header_footer_rows, annotated_board_contents, header_footer_rows))
        details = "\nMove: {}. Captures X: {} O: {}\n".format(
            self.n, *captures)
        return annotated_board + details

    def is_move_suicidal(self, move):
        potential_libs = set()
        for n in NEIGHBORS[move]:
            neighbor_group_id = self.lib_tracker.group_index[n]
            if neighbor_group_id == MISSING_GROUP_ID:
                # at least one liberty after playing here, so not a suicide
                return False
            neighbor_group = self.lib_tracker.groups[neighbor_group_id]
            if neighbor_group.color == self.to_play:
                potential_libs |= neighbor_group.liberties
            elif len(neighbor_group.liberties) == 1:
                # would capture an opponent group if they only had one lib.
                return False
        # it's possible to suicide by connecting several friendly groups
        # each of which had one liberty.
        potential_libs -= {move}
        return not potential_libs

    def is_move_legal(self, move):
        'Checks that a move is on an empty space, not on ko, and not suicide'
        if move is None:
            return True
        if self.board[move] != EMPTY:
            return False
        if move == self.ko:
            return False
        if self.is_move_suicidal(move):
            return False

        return True

    def all_legal_moves(self):
        'Returns a np.array of size go.N**2 + 1, with 1 = legal, 0 = illegal'
        # by default, every move is legal
        legal_moves = np.ones([N, N], dtype=np.int8)
        # ...unless there is already a stone there
        legal_moves[self.board != EMPTY] = 0
        # calculate which spots have 4 stones next to them
        # padding is because the edge always counts as a lost liberty.
        adjacent = np.ones([N + 2, N + 2], dtype=np.int8)
        adjacent[1:-1, 1:-1] = np.abs(self.board)
        num_adjacent_stones = (adjacent[:-2, 1:-1] + adjacent[1:-1, :-2]
                               + adjacent[2:, 1:-1] + adjacent[1:-1, 2:])
        # Surrounded spots are those that are empty and have 4 adjacent stones.
        surrounded_spots = np.multiply(
            (self.board == EMPTY),
            (num_adjacent_stones == 4))
        # Such spots are possibly illegal, unless they are capturing something.
        # Iterate over and manually check each spot.
        for coord in np.transpose(np.nonzero(surrounded_spots)):
            if self.is_move_suicidal(tuple(coord)):
                legal_moves[tuple(coord)] = 0

        # ...and retaking ko is always illegal
        if self.ko is not None:
            legal_moves[self.ko] = 0

        # and pass is always legal
        return np.concatenate([legal_moves.ravel(), [1]])

    def pass_move(self, mutate=False):
        pos = self if mutate else copy.deepcopy(self)
        pos.n += 1
        pos.recent += (PlayerMove(pos.to_play, None),)
        pos.board_deltas = np.concatenate((
            np.zeros([1, N, N], dtype=np.int8),
            pos.board_deltas[:6]))
        pos.to_play *= -1
        pos.ko = None
        return pos

    def flip_playerturn(self, mutate=False):
        pos = self if mutate else copy.deepcopy(self)
        pos.ko = None
        pos.to_play *= -1
        return pos

    def get_liberties(self):
        return self.lib_tracker.liberty_cache

    def play_move(self, c, color=None, mutate=False):
        # Obeys CGOS Rules of Play. In short:
        # No suicides
        # Chinese/area scoring
        # Positional superko (this is very crudely approximate at the moment.)
        if color is None:
            color = self.to_play

        pos = self if mutate else copy.deepcopy(self)

        if c is None:
            pos = pos.pass_move(mutate=mutate)
            return pos

        if not self.is_move_legal(c):
            raise IllegalMove("{} move at {} is illegal: \n{}".format(
                "Black" if self.to_play == BLACK else "White",
                coords.to_gtp(c), self))

        potential_ko = is_koish(self.board, c)

        place_stones(pos.board, color, [c])
        captured_stones = pos.lib_tracker.add_stone(color, c)
        place_stones(pos.board, EMPTY, captured_stones)

        opp_color = color * -1

        new_board_delta = np.zeros([N, N], dtype=np.int8)
        new_board_delta[c] = color
        place_stones(new_board_delta, color, captured_stones)

        if len(captured_stones) == 1 and potential_ko == opp_color:
            new_ko = list(captured_stones)[0]
        else:
            new_ko = None

        if pos.to_play == BLACK:
            new_caps = (pos.caps[0] + len(captured_stones), pos.caps[1])
        else:
            new_caps = (pos.caps[0], pos.caps[1] + len(captured_stones))

        pos.n += 1
        pos.caps = new_caps
        pos.ko = new_ko
        pos.recent += (PlayerMove(color, c),)

        # keep a rolling history of last 7 deltas - that's all we'll need to
        # extract the last 8 board states.
        pos.board_deltas = np.concatenate((
            new_board_delta.reshape(1, N, N),
            pos.board_deltas[:6]))
        pos.to_play *= -1
        return pos

    def is_game_over(self):
        return (len(self.recent) >= 2
                and self.recent[-1].move is None
                and self.recent[-2].move is None)

    def score(self):
        'Return score from B perspective. If W is winning, score is negative.'
        working_board = np.copy(self.board)
        while EMPTY in working_board:
            unassigned_spaces = np.where(working_board == EMPTY)
            c = unassigned_spaces[0][0], unassigned_spaces[1][0]
            territory, borders = find_reached(working_board, c)
            border_colors = {working_board[b] for b in borders}
            X_border = BLACK in border_colors
            O_border = WHITE in border_colors
            if X_border and not O_border:
                territory_color = BLACK
            elif O_border and not X_border:
                territory_color = WHITE
            else:
                territory_color = UNKNOWN  # dame, or seki
            place_stones(working_board, territory_color, territory)

        return np.count_nonzero(working_board == BLACK) - np.count_nonzero(working_board == WHITE) - self.komi

    def result(self):
        score = self.score()
        if score > 0:
            return 1
        elif score < 0:
            return -1
        else:
            return 0

    def result_string(self):
        score = self.score()
        if score > 0:
            return 'B+' + '%.1f' % score
        elif score < 0:
            return 'W+' + '%.1f' % abs(score)
        else:
            return 'DRAW'