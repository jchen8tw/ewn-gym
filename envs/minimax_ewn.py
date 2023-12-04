from envs import EinsteinWuerfeltNichtEnv
from constants import Player
from typing import Tuple, Optional
import numpy as np


class MinimaxEnv(EinsteinWuerfeltNichtEnv):
    """This class is used to evaluate the board state for the minimax algorithm
    """

    def __init__(self, board_size: int = 5,
                 cube_layer: int = 3, seed: int = 9487, reward: float = 1., agent_player: Player = Player.TOP_LEFT, render_mode: Optional[str] = None, opponent_policy: str = "random", **policy_kwargs):
        super().__init__(board_size=board_size,
                         cube_layer=cube_layer, seed=seed, reward=reward, agent_player=agent_player, render_mode=render_mode, opponent_policy=opponent_policy, **policy_kwargs)
        # This is defined in the original env now
        # self.cube_num: int = cube_layer * (cube_layer + 1) // 2

    def get_opponent(self, player: Player):
        return Player.BOTTOM_RIGHT if player == Player.TOP_LEFT else Player.TOP_LEFT

    def set_dice_roll(self, roll: int):
        self.dice_roll = roll

    def evaluate(self) -> float:
        # Check if the agent player has won or lost
        # Used when searching to the end game
        if self.agent_player == Player.TOP_LEFT:
            if self.board[-1, -
                          1] > 0 or not np.any(self.board < 0):  # Agent player wins
                return 10
            if self.board[0, 0] < 0 or not np.any(
                    self.board > 0):  # Agent player loses
                return -10

        elif self.agent_player == Player.BOTTOM_RIGHT:
            if self.board[0, 0] < 0 or not np.any(
                    self.board > 0):  # Agent player wins
                return 10
            if self.board[-1, -
                          1] > 0 or not np.any(self.board < 0):  # Agent player loses
                return -10

        # Check the min distance to the goal and the number of remaining cubes
        # to calculate the value of the board
        board_height, board_width = self.board.shape
        board_length = max(board_height, board_width)
        score = 0
        min_dist_positive, min_dist_negative = float('inf'), float('inf')
        remaining_cubes_positive, remaining_cubes_negative = 0, 0
        for i in range(board_height):
            for j in range(board_width):
                if self.board[i, j] > 0:  # Pieces of TOP_LEFT player
                    curr_dist = max(
                        (board_height - 1 - i),
                        (board_width - 1 - j))
                    min_dist_positive = min(min_dist_positive, curr_dist)
                    remaining_cubes_positive += 1
                elif self.board[i, j] < 0:  # Pieces of BOTTOM_RIGHT player
                    curr_dist = max(
                        (board_height - 1 - i),
                        (board_width - 1 - j))
                    min_dist_negative = min(min_dist_negative, curr_dist)
                    remaining_cubes_negative += 1

        score += (board_length - min_dist_positive) * \
            (1 / remaining_cubes_positive)
        score -= (board_length - min_dist_negative) * \
            (1 / remaining_cubes_negative)

        """
        # Bad evaluation function
        # Calculate the distance of each player's pieces to the opposite corner
        for i in range(board_height):
            for j in range(board_width):
                if self.board[i, j] > 0:  # Pieces of TOP_LEFT player
                    # Add distance from bottom-right corner
                    score += (board_height - 1 - i) + (board_width - 1 - j)
                elif self.board[i, j] < 0:  # Pieces of BOTTOM_RIGHT player
                    # Subtract distance from top-left corner
                    score -= i + j
        # Consider the impact of the number of pieces remaining
        top_left_pieces = np.count_nonzero(self.board > 0)
        bottom_right_pieces = np.count_nonzero(self.board < 0)
        piece_difference = top_left_pieces - bottom_right_pieces
        some_factor = 1  # This is a parameter that might need adjustment
        score += piece_difference * some_factor
        """

        # Return a positive score for TOP_LEFT player, and negative for
        # BOTTOM_RIGHT
        return score if self.agent_player == Player.TOP_LEFT else -score
