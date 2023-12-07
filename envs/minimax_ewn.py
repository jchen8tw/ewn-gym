from envs import EinsteinWuerfeltNichtEnv
from constants import Player
from typing import Tuple, Optional
import numpy as np


class MinimaxEnv(EinsteinWuerfeltNichtEnv):
    """This class is used to evaluate the board state for the minimax algorithm
    """

    def __init__(self, board_size: int = 5,
                 cube_layer: int = 3, seed: int = 9487, reward: float = 1., agent_player: Player = Player.TOP_LEFT, render_mode: Optional[str] = None, opponent_policy: str = "random"):
        super().__init__(board_size=board_size,
                         cube_layer=cube_layer, seed=seed, reward=reward, agent_player=agent_player, render_mode=render_mode, opponent_policy=opponent_policy)
        # This is defined in the original env now
        # self.cube_num: int = cube_layer * (cube_layer + 1) // 2

    def get_opponent(self, player: Player):
        return Player.BOTTOM_RIGHT if player == Player.TOP_LEFT else Player.TOP_LEFT

    def set_dice_roll(self, roll: int):
        self.dice_roll = roll

    def evaluate(self, heuristic='hybrid') -> float:
        if heuristic == 'min_dist':
            return self.evaluate_min_dist();
        elif heuristic == 'two_min_dist':
            return self.evaluate_two_min_dist();
        elif heuristic == 'attk':
            return self.evaluate_attack();

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

        # Return a positive score for TOP_LEFT player, and negative for
        # BOTTOM_RIGHT
        return score if self.agent_player == Player.TOP_LEFT else -score
    
    def evaluate_min_dist(self) -> float:
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
        for i in range(board_height):
            for j in range(board_width):
                if self.board[i, j] > 0:  # Pieces of TOP_LEFT player
                    curr_dist = max(
                        (board_height - 1 - i),
                        (board_width - 1 - j))
                    min_dist_positive = min(min_dist_positive, curr_dist)
                elif self.board[i, j] < 0:  # Pieces of BOTTOM_RIGHT player
                    curr_dist = max(
                        (board_height - 1 - i),
                        (board_width - 1 - j))
                    min_dist_negative = min(min_dist_negative, curr_dist)

        score += (board_length - min_dist_positive)
        score -= (board_length - min_dist_negative)

        # Return a positive score for TOP_LEFT player, and negative for
        # BOTTOM_RIGHT
        return score if self.agent_player == Player.TOP_LEFT else -score
    
    def evaluate_two_min_dist(self) -> float:
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
        distances_positive, distances_negative = [], []

        for i in range(board_height):
            for j in range(board_width):
                if self.board[i, j] > 0:  # Pieces of TOP_LEFT player
                    curr_dist = max((board_height - 1 - i), (board_width - 1 - j))
                    distances_positive.append(curr_dist)
                elif self.board[i, j] < 0:  # Pieces of BOTTOM_RIGHT player
                    curr_dist = max((board_height - 1 - i), (board_width - 1 - j))
                    distances_negative.append(curr_dist)

        # Sort the lists and take the first two minimum distances
        distances_positive.sort()
        distances_negative.sort()
        min_two_dist_positive = distances_positive[:2] if len(distances_positive) >= 2 else distances_positive
        min_two_dist_negative = distances_negative[:2] if len(distances_negative) >= 2 else distances_negative
        
        score += sum(min_two_dist_negative) - sum(min_two_dist_positive)

        # Return a positive score for TOP_LEFT player, and negative for
        # BOTTOM_RIGHT
        return score if self.agent_player == Player.TOP_LEFT else -score
    
    def evaluate_attack(self) -> float:
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
        remaining_cubes_positive, remaining_cubes_negative = 0, 0
        for i in range(board_height):
            for j in range(board_width):
                if self.board[i, j] > 0:  # Pieces of TOP_LEFT player
                    remaining_cubes_positive += 1
                elif self.board[i, j] < 0:  # Pieces of BOTTOM_RIGHT player
                    remaining_cubes_negative += 1

        score = -(remaining_cubes_positive + remaining_cubes_negative)
        return score
