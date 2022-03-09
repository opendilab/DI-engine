import chess
import chess_utils
import numpy as np


def assert_asserts(x):
    try:
        x()
    except AssertionError:
        return True
    return False


assert chess_utils.move_to_coord(chess.Move.from_uci("a8b7")) == (0, 7)
assert chess_utils.move_to_coord(chess.Move.from_uci("g3b7")) == (6, 2)

assert (chess_utils.get_knight_dir((2, 1)) == 7)
assert (chess_utils.get_knight_dir((-2, 1)) == 1)
assert assert_asserts(lambda: chess_utils.get_knight_dir((-1, 1)))

assert chess_utils.get_queen_dir((5, -5)) == (4, 5)
assert chess_utils.get_queen_dir((8, 0)) == (7, 6)
assert chess_utils.get_queen_dir((0, -1)) == (0, 3)
assert assert_asserts(lambda: chess_utils.get_queen_dir((0, 0)))
assert assert_asserts(lambda: chess_utils.get_queen_dir((1, 2)))
assert assert_asserts(lambda: chess_utils.get_queen_dir((2, -8)))

assert chess_utils.get_move_plane(chess.Move.from_uci("e1g1"), chess.KING) == chess_utils.get_queen_plane((2, 0))  # castles kingside
assert chess_utils.get_move_plane(chess.Move.from_uci("g1f3"), chess.KNIGHT) == 56 + chess_utils.get_knight_dir((-1, 2))  # castles kingside
assert chess_utils.get_move_plane(chess.Move.from_uci("f7f8q"), chess.PAWN) == chess_utils.get_queen_plane((0, 1))
assert chess_utils.get_move_plane(chess.Move.from_uci("f7f8r"), chess.PAWN) == 56 + 8 + 2 + 1 * 3
assert chess_utils.get_move_plane(chess.Move.from_uci("f7g8n"), chess.PAWN) == 56 + 8 + 0 + 2 * 3

assert str(chess_utils.mirror_move(chess.Move.from_uci("f7g8"))) == "f2g1"

board = chess.Board()
board.push_san("e4")
print(chess_utils.sample_action(board, np.ones([8, 8, 73])))
print(chess_utils.sample_action(board, np.ones([8, 8, 73])))
test_action = np.ones([8, 8, 73]) * -100
test_action[0, 1, 4] = 1
assert str(chess_utils.sample_action(board, test_action)) == "a2a4"
board.push_san("c5")
obs = chess_utils.get_observation(board, player=1)
board.push_san("e5")
obs = chess_utils.get_observation(board, player=1)
board.push_san("d5")
obs = chess_utils.get_observation(board, player=1)
board.push_san("a3")
obs = chess_utils.get_observation(board, player=1)
board.push_san("d4")
obs = chess_utils.get_observation(board, player=1)
board.push_san("c4")
obs = chess_utils.get_observation(board, player=1)
