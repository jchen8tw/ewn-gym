from envs import MinimaxEnv
from constants import Player, ClassicalPolicy
from typing import Tuple, Optional
import numpy as np

"""
This file contains the environment for training the agent using all kinds of heuristic
"""


class MiniMaxHeuristicEnv(MinimaxEnv):
    """
    This class uses the heuristic function of minimax to evaluate the board.
    By leveraging the heuristic function, we can calculate the TD value between
    two states for reward shaping.
    The score of evaluating the board is between -10 and 10.
    """

    def __init__(self, board_size: int = 5,
                 cube_layer: int = 3,
                 seed: int = 9487,
                 goal_reward: float = 10.,
                 agent_player: Player = Player.TOP_LEFT,
                 render_mode: Optional[str] = None,
                 opponent_policy: ClassicalPolicy | str = ClassicalPolicy.random,
                 # These two parameters are for reward shaping
                 illegal_move_reward: float = -1.0,
                 illegal_move_tolerance: int = 10,
                 **policy_kwargs):
        # note that the original env reward is the goal reward
        super().__init__(board_size=board_size,
                         cube_layer=cube_layer, seed=seed, reward=goal_reward, agent_player=agent_player, render_mode=render_mode, opponent_policy=opponent_policy, **policy_kwargs)

        # The previous score of the agent player
        self.prev_score: float = self.evaluate()
        self.illegal_move_reward: float = illegal_move_reward
        # The number of illegal moves the agent can make before the game ends
        self.illegal_move_tolerance: int = illegal_move_tolerance

    def step(self, action: np.ndarray):
        # Determine the cube to move based on the dice roll
        cube_to_move_index = self.find_cube_to_move(action[0] == 1)

        # Execute the move
        valid: bool = self.execute_move(cube_to_move_index, action[1])

        # Check if the move is valid
        if not valid:
            self.illegal_move_tolerance -= 1
            if self.illegal_move_tolerance <= 0:
                return {"board": self.board,
                        "dice_roll": self.dice_roll}, -self.reward, True, True, {
                    "message": "Invalid move for player! End the game."}
            return {"board": self.board,
                    "dice_roll": self.dice_roll}, self.illegal_move_reward, False, False, {
                "message": f"Invalid move for player! Tolerance left {self.illegal_move_tolerance}."}

        # Check for win condition
        if self.check_win():
            return {"board": self.board,
                    "dice_roll": self.dice_roll}, self.reward, True, False, {
                "message": "You won!"}

        # Switch turn
        self.switch_player()
        self.roll_dice()  # Roll the dice for opponent's turn

        # Perform opponent's action
        opponent_action: np.ndarray = self.opponent_action()

        # Determine the cube to move based on the dice roll
        cube_to_move_index = self.find_cube_to_move(opponent_action[0] == 1)

        # Execute the move
        valid = self.execute_move(cube_to_move_index, opponent_action[1])

        # Check if the move is valid
        if not valid:
            return {"board": self.board,
                    "dice_roll": self.dice_roll}, 0, True, True, {
                "message": "Invalid move for opponent! End the game."}

        if self.check_win():
            return {"board": self.board,
                    "dice_roll": self.dice_roll}, -self.reward, True, False, {
                "message": "You lost!"}

        # Switch turn
        self.switch_player()
        # Roll the dice for the next turn
        self.roll_dice()

        # Calculate the reward
        current_board_score: float = self.evaluate()
        reward: float = current_board_score - self.prev_score
        self.prev_score = current_board_score

        return {"board": self.board,
                "dice_roll": self.dice_roll}, reward, False, False, {}
