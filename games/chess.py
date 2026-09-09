import datetime
import pathlib
import random
from collections import deque

import chess
import numpy
import torch

from .abstract_game import AbstractGame


# 64 source squares × 64 destination squares × 5 promotion choices.
#
# Promotion choices:
#   0: no promotion
#   1: knight
#   2: bishop
#   3: rook
#   4: queen
ACTION_SPACE_SIZE = 64 * 64 * 5

PROMOTION_TO_INDEX = {
    None: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
}

INDEX_TO_PROMOTION = {
    0: None,
    1: chess.KNIGHT,
    2: chess.BISHOP,
    3: chess.ROOK,
    4: chess.QUEEN,
}


def move_to_action(move):
    """
    Convert a python-chess Move to a fixed MuZero action integer.
    """
    if move.promotion not in PROMOTION_TO_INDEX:
        raise ValueError(
            f"Unsupported promotion piece: {move.promotion}"
        )

    promotion_index = PROMOTION_TO_INDEX[move.promotion]

    return (
        (move.from_square * 64 + move.to_square) * 5
        + promotion_index
    )


def action_to_move(action):
    """
    Convert a fixed MuZero action integer to a python-chess Move.
    """
    action = int(action)

    if action < 0 or action >= ACTION_SPACE_SIZE:
        raise ValueError(f"Invalid chess action: {action}")

    promotion_index = action % 5
    action //= 5

    to_square = action % 64
    from_square = action // 64

    return chess.Move(
        from_square,
        to_square,
        promotion=INDEX_TO_PROMOTION[promotion_index],
    )


def square_to_coordinates(square, perspective):
    """
    Convert a chess square to row and column coordinates.

    The returned coordinates are from the perspective of the player
    whose turn it is.
    """
    file_index = chess.square_file(square)
    rank_index = chess.square_rank(square)

    if perspective == chess.WHITE:
        row = 7 - rank_index
        column = file_index
    else:
        row = rank_index
        column = 7 - file_index

    return row, column


def encode_position(board):
    """
    Encode a board using an AlphaZero-style 119-plane representation.

    Plane layout:

        0-5:
            Current player's pawn, knight, bishop, rook, queen, king

        6-11:
            Opponent's pawn, knight, bishop, rook, queen, king

        12-107:
            Eight historical positions, each represented by 12 piece planes

        108:
            Side to play

        109:
            Current player's kingside castling rights

        110:
            Current player's queenside castling rights

        111:
            Opponent's kingside castling rights

        112:
            Opponent's queenside castling rights

        113:
            En-passant target square

        114:
            Halfmove clock

        115:
            Fullmove number

        116:
            Current position is a repetition

        117:
            Current player is in check

        118:
            Legal move count

    This function encodes only the supplied position. Historical positions
    are inserted by Chess.get_observation().
    """
    observation = numpy.zeros(
        (119, 8, 8),
        dtype=numpy.float32,
    )

    current_player = board.turn
    opponent = not current_player

    piece_types = [
        chess.PAWN,
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
        chess.KING,
    ]

    def add_piece_planes(color, plane_offset):
        for piece_index, piece_type in enumerate(piece_types):
            for square in board.pieces(piece_type, color):
                row, column = square_to_coordinates(
                    square,
                    current_player,
                )
                observation[
                    plane_offset + piece_index,
                    row,
                    column,
                ] = 1.0

    add_piece_planes(current_player, 0)
    add_piece_planes(opponent, 6)

    # The observation is canonicalized to the player to move.
    observation[108, :, :] = 1.0

    if current_player == chess.WHITE:
        current_kingside = board.has_kingside_castling_rights(
            chess.WHITE
        )
        current_queenside = board.has_queenside_castling_rights(
            chess.WHITE
        )
        opponent_kingside = board.has_kingside_castling_rights(
            chess.BLACK
        )
        opponent_queenside = board.has_queenside_castling_rights(
            chess.BLACK
        )
    else:
        current_kingside = board.has_kingside_castling_rights(
            chess.BLACK
        )
        current_queenside = board.has_queenside_castling_rights(
            chess.BLACK
        )
        opponent_kingside = board.has_kingside_castling_rights(
            chess.WHITE
        )
        opponent_queenside = board.has_queenside_castling_rights(
            chess.WHITE
        )

    if current_kingside:
        observation[109, :, :] = 1.0

    if current_queenside:
        observation[110, :, :] = 1.0

    if opponent_kingside:
        observation[111, :, :] = 1.0

    if opponent_queenside:
        observation[112, :, :] = 1.0

    if board.ep_square is not None:
        row, column = square_to_coordinates(
            board.ep_square,
            current_player,
        )
        observation[113, row, column] = 1.0

    observation[114, :, :] = min(
        board.halfmove_clock / 100.0,
        1.0,
    )

    observation[115, :, :] = min(
        board.fullmove_number / 200.0,
        1.0,
    )

    if board.is_repetition(2):
        observation[116, :, :] = 1.0

    if board.is_check():
        observation[117, :, :] = 1.0

    legal_move_count = board.legal_moves.count()
    observation[118, :, :] = min(
        legal_move_count / 218.0,
        1.0,
    )

    return observation


class MuZeroConfig:
    def __init__(self):
        # General
        self.seed = 0
        self.max_num_gpus = 1
        self.ray_num_cpus = 8
        self.cpu_actor_num_cpus = 1
        self.trainer_num_cpus = 2
        self.selfplay_num_cpus = 2
        self.test_num_cpus = 1
        self.reanalyse_num_cpus = 1
        self.replay_buffer_num_cpus = 1
        self.shared_storage_num_cpus = 1

        # Game
        self.observation_shape = (119, 8, 8)
        self.action_space = list(range(ACTION_SPACE_SIZE))
        self.players = [0, 1]
        self.stacked_observations = 0

        # Evaluation
        self.muzero_player = 0
        self.opponent = "expert"

        # Self-play
        self.num_workers = 4
        self.selfplay_on_gpu = False
        self.max_moves = 128
        self.num_simulations = 10
        self.discount = 0.997
        self.temperature_threshold = None

        # MCTS exploration
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # Network
        self.network = "resnet"
        self.support_size = 10
        self.downsample = False

        self.blocks = 2
        self.channels = 32

        self.reduced_channels_reward = 8
        self.reduced_channels_value = 8
        self.reduced_channels_policy = 8

        self.resnet_fc_reward_layers = [64]
        self.resnet_fc_value_layers = [64]
        self.resnet_fc_policy_layers = [64]

        # Required unused fully-connected settings
        self.encoding_size = 32
        self.fc_representation_layers = []
        self.fc_dynamics_layers = [64]
        self.fc_reward_layers = [64]
        self.fc_value_layers = [64]
        self.fc_policy_layers = [64]

        # Training
        self.results_path = (
            pathlib.Path(__file__).resolve().parents[1]
            / "results"
            / pathlib.Path(__file__).stem
            / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        )

        self.save_model = True
        self.training_steps = 5000
        self.batch_size = 16
        self.checkpoint_interval = 100
        self.value_loss_weight = 0.25
        self.train_on_gpu = True

        self.optimizer = "Adam"
        self.weight_decay = 1e-4
        self.momentum = 0.9

        self.lr_init = 0.0003
        self.lr_decay_rate = 1.0
        self.lr_decay_steps = 10000

        # Replay
        self.replay_buffer_size = 500
        self.num_unroll_steps = 5
        self.td_steps = 20
        self.PER = True
        self.PER_alpha = 0.5

        # Reanalyse
        self.use_last_model_value = False
        self.reanalyse_on_gpu = False

        # Ratio
        self.self_play_delay = 0
        self.training_delay = 0
        self.ratio = 2.0

    def visit_softmax_temperature_fn(self, trained_steps):
        # Use stochastic search throughout initial experiments.
        return 1.0


class Game(AbstractGame):
    """
    MuZero wrapper around the chess environment.
    """

    def __init__(self, seed=None, render_mode=None):
        self.env = Chess(seed=seed)

    def step(self, action):
        """
        Apply an action.

        Returns:
            observation, reward, done
        """
        observation, reward, done = self.env.step(action)

        # The Tic-Tac-Toe example scales rewards by 20.
        # Chess does not need that scaling because the result is already
        # represented as -1, 0, or +1.
        return observation, reward, done

    def to_play(self):
        return self.env.to_play()

    def legal_actions(self):
        return self.env.legal_actions()

    def reset(self):
        return self.env.reset()

    def close(self):
        self.env.close()

    def render(self):
        self.env.render()
        input("Press enter to take a step ")

    def human_to_action(self):
        """
        Ask for a move in UCI notation, for example e2e4 or g1f3.
        """
        legal_moves = list(self.env.board.legal_moves)

        print(
            "Legal moves:",
            " ".join(move.uci() for move in legal_moves),
        )

        while True:
            move_text = input(
                "Enter a legal move in UCI notation: "
            ).strip()

            try:
                move = chess.Move.from_uci(move_text)
            except ValueError:
                print("Invalid UCI move.")
                continue

            if move in legal_moves:
                return move_to_action(move)

            print("Illegal move.")

    def expert_agent(self):
        return self.env.expert_action()

    def action_to_string(self, action_number):
        try:
            return action_to_move(action_number).uci()
        except ValueError:
            return f"Invalid action: {action_number}"


class Chess:
    """
    Chess environment using python-chess.

    Player 0 is White.
    Player 1 is Black.
    """

    def __init__(self, seed=None, render_mode=None):
        self.seed = seed
        self.render_mode = render_mode

        if seed is not None:
            random.seed(seed)
            numpy.random.seed(seed)

        self.board = chess.Board()

        # Store positions from oldest to newest.
        # The current position is included as the last item.
        self.position_history = deque(maxlen=8)
        self.position_history.append(self.board.copy(stack=True))

    def to_play(self):
        """
        Return 0 for White and 1 for Black.
        """
        return 0 if self.board.turn == chess.WHITE else 1

    def reset(self):
        self.board.reset()
        self.position_history.clear()
        self.position_history.append(self.board.copy(stack=True))
        return self.get_observation()

    def step(self, action):
        """
        Apply a legal action.

        Rewards are from the perspective of the player who made the move:

            +1: the move won the game
             0: draw or nonterminal move
            -1: defensive fallback for an invalid terminal result
        """
        move = action_to_move(action)

        if move not in self.board.legal_moves:
            raise ValueError(
                f"Illegal chess action {action}: {move.uci()}"
            )

        player_who_moved = self.to_play()
        self.board.push(move)

        self.position_history.append(self.board.copy(stack=True))

        outcome = self.board.outcome(claim_draw=True)

        if outcome is None:
            reward = 0.0
            done = False
        elif outcome.winner is None:
            reward = 0.0
            done = True
        else:
            winner = 0 if outcome.winner == chess.WHITE else 1
            reward = 1.0 if winner == player_who_moved else -1.0
            done = True

        return self.get_observation(), reward, done

    def get_observation(self):
        """
        Return a canonical AlphaZero-style observation.

        The current board is represented in planes 0-118. Previous
        positions are stored in planes 12-107, with the newest previous
        position closest to the current position.

        The current position's metadata planes are retained.
        """
        observation = numpy.zeros(
            (119, 8, 8),
            dtype=numpy.float32,
        )

        current = self.position_history[-1]
        current_encoded = encode_position(current)

        # Current piece planes.
        observation[0:12] = current_encoded[0:12]

        # Historical piece planes.
        previous_positions = list(self.position_history)[:-1]
        previous_positions = previous_positions[-8:]

        # Most recent previous position occupies planes 12:24.
        for index, historical_board in enumerate(
            reversed(previous_positions)
        ):
            encoded = encode_position(historical_board)
            start = 12 + index * 12
            end = start + 12

            if end <= 108:
                observation[start:end] = encoded[0:12]

        # Current position metadata.
        observation[108:119] = current_encoded[108:119]

        return observation

    def legal_actions(self):
        """
        Return all legal chess actions.
        """
        return [
            move_to_action(move)
            for move in self.board.legal_moves
        ]

    def expert_action(self):
        """
        Simple legal heuristic agent.

        Priority:
            1. Checkmate in one
            2. Capture the highest-value piece
            3. Promote a pawn
            4. Random legal move

        This is only a baseline opponent, not a chess engine.
        """
        legal_moves = list(self.board.legal_moves)

        if not legal_moves:
            raise RuntimeError("No legal actions available.")

        # Prefer immediate checkmate.
        for move in legal_moves:
            board = self.board.copy(stack=True)
            board.push(move)

            if board.is_checkmate():
                return move_to_action(move)

        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 100,
        }

        best_score = None
        best_moves = []

        for move in legal_moves:
            score = 0

            if move.promotion is not None:
                score += 8

            captured_piece = self.board.piece_at(move.to_square)

            if captured_piece is not None:
                score += piece_values[captured_piece.piece_type]

            board = self.board.copy(stack=True)
            board.push(move)

            if board.is_check():
                score += 2

            if best_score is None or score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)

        return move_to_action(
            random.choice(best_moves)
        )

    def render(self):
        print()
        print(self.board)
        print()

        player = "White" if self.board.turn == chess.WHITE else "Black"

        print(f"Player to move: {player}")
        print(f"FEN: {self.board.fen()}")
        print(f"Legal moves: {self.board.legal_moves.count()}")

        outcome = self.board.outcome(claim_draw=True)

        if outcome is not None:
            if outcome.winner is None:
                print("Result: draw")
            elif outcome.winner == chess.WHITE:
                print("Result: White wins")
            else:
                print("Result: Black wins")
