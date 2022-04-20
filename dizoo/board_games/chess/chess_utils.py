import chess
import numpy as np


def boards_to_ndarray(boards):
    arr64 = np.array(boards, dtype=np.uint64)
    arr8 = arr64.view(dtype=np.uint8)
    bits = np.unpackbits(arr8)
    floats = bits.astype(bool)
    boardstack = floats.reshape([len(boards), 8, 8])
    boardimage = np.transpose(boardstack, [1, 2, 0])
    return boardimage


def square_to_coord(s):
    col = s % 8
    row = s // 8
    return (col, row)


def diff(c1, c2):
    x1, y1 = c1
    x2, y2 = c2
    return (x2 - x1, y2 - y1)


def sign(v):
    return -1 if v < 0 else (1 if v > 0 else 0)


def mirror_move(move):
    return chess.Move(chess.square_mirror(move.from_square), chess.square_mirror(move.to_square), promotion=move.promotion)


def result_to_int(result_str):
    if result_str == "1-0":
        return 1
    elif result_str == "0-1":
        return -1
    elif result_str == "1/2-1/2":
        return 0
    else:
        assert False, "bad result"


def get_queen_dir(diff):
    dx, dy = diff
    assert dx == 0 or dy == 0 or abs(dx) == abs(dy)
    magnitude = max(abs(dx), abs(dy)) - 1

    assert magnitude < 8 and magnitude >= 0
    counter = 0
    for x in range(-1, 1 + 1):
        for y in range(-1, 1 + 1):
            if x == 0 and y == 0:
                continue
            if x == sign(dx) and y == sign(dy):
                return magnitude, counter
            counter += 1
    assert False, "bad queen move inputted"


def get_queen_plane(diff):
    NUM_COUNTERS = 8
    mag, counter = get_queen_dir(diff)
    return mag * NUM_COUNTERS + counter


def get_knight_dir(diff):
    dx, dy = diff
    counter = 0
    for x in range(-2, 2 + 1):
        for y in range(-2, 2 + 1):
            if abs(x) + abs(y) == 3:
                if dx == x and dy == y:
                    return counter
                counter += 1
    assert False, "bad knight move inputted"


def is_knight_move(diff):
    dx, dy = diff
    return abs(dx) + abs(dy) == 3 and 1 <= abs(dx) <= 2


def get_pawn_promotion_move(diff):
    dx, dy = diff
    assert dy == 1
    assert -1 <= dx <= 1
    return dx + 1


def get_pawn_promotion_num(promotion):
    assert promotion == chess.KNIGHT or promotion == chess.BISHOP or promotion == chess.ROOK
    return 0 if promotion == chess.KNIGHT else (1 if promotion == chess.BISHOP else 2)


def move_to_coord(move):
    return square_to_coord(move.from_square)


def get_move_plane(move):
    source = move.from_square
    dest = move.to_square
    difference = diff(square_to_coord(source), square_to_coord(dest))

    QUEEN_MOVES = 56
    KNIGHT_MOVES = 8
    QUEEN_OFFSET = 0
    KNIGHT_OFFSET = QUEEN_MOVES
    UNDER_OFFSET = KNIGHT_OFFSET + KNIGHT_MOVES

    if is_knight_move(difference):
        return KNIGHT_OFFSET + get_knight_dir(difference)
    else:
        if move.promotion is not None and move.promotion != chess.QUEEN:
            return UNDER_OFFSET + 3 * get_pawn_promotion_move(difference) + get_pawn_promotion_num(move.promotion)
        else:
            return QUEEN_OFFSET + get_queen_plane(difference)


moves_to_actions = {}
actions_to_moves = {}


def action_to_move(board, action, player):
    base_move = chess.Move.from_uci(actions_to_moves[action])

    base_coord = square_to_coord(base_move.from_square)
    mirr_move = mirror_move(base_move) if player else base_move
    if mirr_move.promotion == chess.QUEEN:
        mirr_move.promotion = None
    if mirr_move.promotion is None and str(board.piece_at(mirr_move.from_square)).lower() == 'p' and base_coord[1] == 6:
        mirr_move.promotion = chess.QUEEN
    return mirr_move


def make_move_mapping(uci_move):
    TOTAL = 73
    move = chess.Move.from_uci(uci_move)
    source = move.from_square

    coord = square_to_coord(source)
    panel = get_move_plane(move)
    cur_action = (coord[0] * 8 + coord[1]) * TOTAL + panel

    moves_to_actions[uci_move] = cur_action
    actions_to_moves[cur_action] = uci_move


def legal_moves(orig_board):
    '''
    action space is a 8x8x73 dimensional array
    Each of the 8×8
    positions identifies the square from which to “pick up” a piece. The first 56 planes encode
    possible ‘queen moves’ for any piece: a number of squares [1..7] in which the piece will be
    moved, along one of eight relative compass directions {N, NE, E, SE, S, SW, W, NW}. The
    next 8 planes encode possible knight moves for that piece. The final 9 planes encode possible
    underpromotions for pawn moves or captures in two possible diagonals, to knight, bishop or
    rook respectively. Other pawn moves or captures from the seventh rank are promoted to a
    queen
    '''
    if orig_board.turn == chess.BLACK:  # white is 1, black is 0
        board = orig_board.mirror()
    else:
        board = orig_board

    legal_moves = []
    for move in board.legal_moves:
        uci_move = move.uci()
        if uci_move in moves_to_actions:
            legal_moves.append(moves_to_actions[move.uci()])
        else:
            make_move_mapping(uci_move)
            legal_moves.append(moves_to_actions[move.uci()])

    return legal_moves


def get_observation(orig_board, player):
    '''
    Observation is an 8x8x(P + L) dimensional array
    P is going to be your pieces positions + your opponents pieces positions
    L is going to be some metadata such as repetition count,,
    '''
    board = orig_board
    if player:
        board = board.mirror()
    else:
        board = board

    all_squares = chess.SquareSet(chess.BB_ALL)
    HISTORY_LEN = 1
    PLANES_PER_BOARD = 13
    AUX_SIZE = 7
    RESULT_SIZE = AUX_SIZE + HISTORY_LEN * PLANES_PER_BOARD
    result = [chess.SquareSet(chess.BB_EMPTY) for _ in range(RESULT_SIZE)]
    AUX_OFF = 0
    BASE = AUX_SIZE

    '''        // "Legacy" input planes with:
    // - Plane 104 (0-based) filled with 1 if white can castle queenside.
    // - Plane 105 filled with ones if white can castle kingside.
    // - Plane 106 filled with ones if black can castle queenside.
    // - Plane 107 filled with ones if white can castle kingside.
    if (board.castlings().we_can_000()) result[kAuxPlaneBase + 0].SetAll();
    if (board.castlings().we_can_00()) result[kAuxPlaneBase + 1].SetAll();
    if (board.castlings().they_can_000()) {
      result[kAuxPlaneBase + 2].SetAll();
    }
    if (board.castlings().they_can_00()) result[kAuxPlaneBase + 3].SetAll();
    '''
    if board.castling_rights & chess.BB_H1:
        result[AUX_OFF + 0] = all_squares
    if board.castling_rights & chess.BB_A1:
        result[AUX_OFF + 1] = all_squares
    if board.castling_rights & chess.BB_H8:
        result[AUX_OFF + 2] = all_squares
    if board.castling_rights & chess.BB_A8:
        result[AUX_OFF + 3] = all_squares
    '''
        if (we_are_black) result[kAuxPlaneBase + 4].SetAll();
        result[kAuxPlaneBase + 5].Fill(history.Last().GetNoCaptureNoPawnPly());
        // Plane kAuxPlaneBase + 6 used to be movecount plane, now it's all zeros.
        // Plane kAuxPlaneBase + 7 is all ones to help NN find board edges.
        result[kAuxPlaneBase + 7].SetAll();
      }
      '''
    if player:
        result[AUX_OFF + 4] = all_squares
    result[AUX_OFF + 5].add(board.halfmove_clock // 2)
    result[AUX_OFF + 6] = all_squares
    '''
      bool flip = false;
      int history_idx = history.GetLength() - 1;
      for (int i = 0; i < std::min(history_planes, kMoveHistory);
           ++i, --history_idx) {
        const Position& position =
            history.GetPositionAt(history_idx < 0 ? 0 : history_idx);
        const ChessBoard& board =
            flip ? position.GetThemBoard() : position.GetBoard();
        if (history_idx < 0 && fill_empty_history == FillEmptyHistory::NO) break;
        // Board may be flipped so compare with position.GetBoard().
        if (history_idx < 0 && fill_empty_history == FillEmptyHistory::FEN_ONLY &&
            position.GetBoard() == ChessBoard::kStartposBoard) {
          break;
        }

        const int base = i * kPlanesPerBoard;
        result[base + 0].mask = (board.ours() & board.pawns()).as_int();
        result[base + 1].mask = (board.our_knights()).as_int();
        result[base + 2].mask = (board.ours() & board.bishops()).as_int();
        result[base + 3].mask = (board.ours() & board.rooks()).as_int();
        result[base + 4].mask = (board.ours() & board.queens()).as_int();
        result[base + 5].mask = (board.our_king()).as_int();

        result[base + 6].mask = (board.theirs() & board.pawns()).as_int();
        result[base + 7].mask = (board.their_knights()).as_int();
        result[base + 8].mask = (board.theirs() & board.bishops()).as_int();
        result[base + 9].mask = (board.theirs() & board.rooks()).as_int();
        result[base + 10].mask = (board.theirs() & board.queens()).as_int();
        result[base + 11].mask = (board.their_king()).as_int();

        '''
    base = BASE
    OURS = 0
    THEIRS = 1
    result[base + 0] = board.pieces(chess.PAWN, OURS)
    result[base + 1] = board.pieces(chess.KNIGHT, OURS)
    result[base + 2] = board.pieces(chess.BISHOP, OURS)
    result[base + 3] = board.pieces(chess.ROOK, OURS)
    result[base + 4] = board.pieces(chess.QUEEN, OURS)
    result[base + 5] = board.pieces(chess.KING, OURS)

    result[base + 6] = board.pieces(chess.PAWN, THEIRS)
    result[base + 7] = board.pieces(chess.KNIGHT, THEIRS)
    result[base + 8] = board.pieces(chess.BISHOP, THEIRS)
    result[base + 9] = board.pieces(chess.ROOK, THEIRS)
    result[base + 10] = board.pieces(chess.QUEEN, THEIRS)
    result[base + 11] = board.pieces(chess.KING, THEIRS)

    '''
    const int repetitions = position.GetRepetitions();
    if (repetitions >= 1) result[base + 12].SetAll();
    '''
    has_repeated = board.is_repetition(2)
    if has_repeated >= 1:
        result[base + 12] = all_squares
    '''
        // If en passant flag is set, undo last pawn move by removing the pawn from
        // the new square and putting into pre-move square.
        if (history_idx < 0 && !board.en_passant().empty()) {
          const auto idx = GetLowestBit(board.en_passant().as_int());
          if (idx < 8) {  // "Us" board
            result[base + 0].mask +=
                ((0x0000000000000100ULL - 0x0000000001000000ULL) << idx);
          } else {
            result[base + 6].mask +=
                ((0x0001000000000000ULL - 0x0000000100000000ULL) << (idx - 56));
          }
        }
        if (history_idx > 0) flip = !flip;
      }
    '''
    # from 0-63
    square = board.ep_square
    if square:
        ours = square > 32
        row = square % 8
        dest_col_add = 8 * 7 if ours else 0
        dest_square = dest_col_add + row
        if ours:
            result[base + 0].remove(square - 8)
            result[base + 0].add(dest_square)
        else:
            result[base + 6].remove(square + 8)
            result[base + 6].add(dest_square)

    return boards_to_ndarray(result)
