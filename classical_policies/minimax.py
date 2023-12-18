from constants import Player
from typing import Dict
import numpy as np
from .base import PolicyBase
from .alpha_zero.ewn.pytorch.NNet import NNetWrapper
from .alpha_zero.ewn.EWNGame import EWNGame


class ExpectiMinimaxAgent(PolicyBase):
    def __init__(self, max_depth: int, cube_layer: int,
                 board_size: int, heuristic='hybrid', **kwargs):
        from envs import MinimaxEnv
        self.max_depth = max_depth
        self.env = MinimaxEnv(
            cube_layer=cube_layer,
            board_size=board_size)
        self.heuristic = heuristic

    def expectiminimax(self, depth: int, player: Player,
                       parent: Player | None, alpha: int, beta: int):

        if self.env.check_win() or depth == 0:
            return self.env.evaluate(heuristic=self.heuristic), None

        # Maximizing player
        if player == self.env.agent_player:
            best_val: float = -float('inf')
            best_action = None
            legal_actions = self.env.get_legal_actions(player)
            curr_dice_roll = self.env.dice_roll
            for action in legal_actions:
                self.env.set_dice_roll(curr_dice_roll)
                self.env.make_simulated_action(player, action)
                val, _ = self.expectiminimax(
                    depth - 1, Player.CHANCE, player, alpha, beta)
                self.env.undo_simulated_action()
                if val > best_val:
                    best_val = val
                    best_action = action
                alpha = max(alpha, best_val)
                if beta <= alpha:
                    break
            return best_val, best_action
        # Minimizing player
        elif player == Player.get_opponent(self.env.agent_player):
            worst_val = float('inf')
            worst_action = None
            legal_actions = self.env.get_legal_actions(player)
            curr_dice_roll = self.env.dice_roll
            for action in legal_actions:
                self.env.set_dice_roll(curr_dice_roll)
                self.env.make_simulated_action(player, action)
                val, _ = self.expectiminimax(
                    depth - 1, Player.CHANCE, player, alpha, beta)
                self.env.undo_simulated_action()
                if val < worst_val:
                    worst_val = val
                    worst_move = action
                beta = min(beta, worst_val)
                if beta <= alpha:
                    break
            return worst_val, worst_action
        # Chance node
        elif player == Player.CHANCE:
            expected_val = 0
            next_player = self.env.agent_player if parent == Player.get_opponent(
                self.env.agent_player) else Player.get_opponent(self.env.agent_player)
            for dice_roll in range(1, 7):
                self.env.set_dice_roll(dice_roll)
                val, _ = self.expectiminimax(
                    depth - 1, next_player, player, alpha, beta)
                expected_val += val / 6
            return expected_val, None

    def restore_env_with_obs(self, obs):
        self.env.dice_roll = obs['dice_roll']
        self.env.board[:] = obs['board']

        for i in range(self.env.cube_num * 2):
            self.env.cube_pos[i] = np.ma.masked
        for i in range(self.env.board.shape[0]):
            for j in range(self.env.board.shape[1]):
                cube = self.env.board[i, j]
                if cube > 0:
                    self.env.cube_pos[cube - 1] = (i, j)
                elif cube < 0:
                    self.env.cube_pos[cube] = (i, j)

    def predict(self, obs, **kwargs):
        self.restore_env_with_obs(obs)
        _, chosen_action = self.expectiminimax(
            self.max_depth, self.env.agent_player, None, -float('inf'), float('inf'))
        return chosen_action, None


class AlphaZeroMinimaxAgent(PolicyBase):
    """Using alpha zero's neural network to predict the value of current state
    """

    def __init__(self, max_depth: int, cube_layer: int,
                 board_size: int,
                 model_folder: str = 'alpha_zero_models',
                 model_name: str = 'checkpoint_242.pth.tar',
                 **policy_kwargs):
        self.max_depth = max_depth
        self.board_size = board_size
        self.cube_layer = cube_layer
        from envs import MinimaxEnv

        self.env = MinimaxEnv(
            cube_layer=cube_layer,
            board_size=board_size)
        game = EWNGame(board_size, cube_layer)
        self.nnet = NNetWrapper(game)
        self.nnet.load_checkpoint(
            folder=model_folder,
            filename=model_name
        )

    def ewn_obs_one_hot(
            self, obs: Dict[str, np.ndarray | int]) -> Dict[str, np.ndarray]:
        """ convert obs to one hot encoding
        """
        cube_num: int = self.cube_layer * (self.cube_layer + 1) // 2
        board = np.zeros(
            (self.board_size,
             self.board_size,
             cube_num),
            dtype=int)
        for i in range(self.board_size):
            for j in range(self.board_size):
                if obs["board"][i, j] > 0:
                    board[i, j, obs["board"][i, j] - 1] = 1
                elif obs["board"][i, j] < 0:
                    board[i, j, -obs["board"][i, j] - 1] = -1

        dice_roll = np.zeros(cube_num, dtype=int)
        dice_roll[obs["dice_roll"] - 1] = 1
        return {"board": board, "dice_roll": dice_roll}

    def evaluate(self) -> float:
        obs: Dict[str, np.ndarray] = {
            "board": self.env.board,
            "dice_roll": self.env.dice_roll
        }
        pi, v = self.nnet.predict(self.ewn_obs_one_hot(obs))
        return v

    def expectiminimax(self, depth: int, player: Player,
                       parent: Player | None, alpha: int, beta: int):

        if self.env.check_win() or depth == 0:
            return self.evaluate(), None

        # Maximizing player
        if player == self.env.agent_player:
            best_val: float = -float('inf')
            best_action = None
            legal_actions = self.env.get_legal_actions(player)
            curr_dice_roll = self.env.dice_roll
            for action in legal_actions:
                self.env.set_dice_roll(curr_dice_roll)
                self.env.make_simulated_action(player, action)
                val, _ = self.expectiminimax(
                    depth - 1, Player.CHANCE, player, alpha, beta)
                self.env.undo_simulated_action()
                if val > best_val:
                    best_val = val
                    best_action = action
                alpha = max(alpha, best_val)
                if beta <= alpha:
                    break
            return best_val, best_action
        # Minimizing player
        elif player == Player.get_opponent(self.env.agent_player):
            worst_val = float('inf')
            worst_action = None
            legal_actions = self.env.get_legal_actions(player)
            curr_dice_roll = self.env.dice_roll
            for action in legal_actions:
                self.env.set_dice_roll(curr_dice_roll)
                self.env.make_simulated_action(player, action)
                val, _ = self.expectiminimax(
                    depth - 1, Player.CHANCE, player, alpha, beta)
                self.env.undo_simulated_action()
                if val < worst_val:
                    worst_val = val
                    worst_move = action
                beta = min(beta, worst_val)
                if beta <= alpha:
                    break
            return worst_val, worst_action
        # Chance node
        elif player == Player.CHANCE:
            expected_val = 0
            next_player = self.env.agent_player if parent == Player.get_opponent(
                self.env.agent_player) else Player.get_opponent(self.env.agent_player)
            for dice_roll in range(1, 7):
                self.env.set_dice_roll(dice_roll)
                val, _ = self.expectiminimax(
                    depth - 1, next_player, player, alpha, beta)
                expected_val += val / 6
            return expected_val, None

    def restore_env_with_obs(self, obs):
        self.env.dice_roll = obs['dice_roll']
        self.env.board[:] = obs['board']

        for i in range(self.env.cube_num * 2):
            self.env.cube_pos[i] = np.ma.masked
        for i in range(self.env.board.shape[0]):
            for j in range(self.env.board.shape[1]):
                cube = self.env.board[i, j]
                if cube > 0:
                    self.env.cube_pos[cube - 1] = (i, j)
                elif cube < 0:
                    self.env.cube_pos[cube] = (i, j)

    def predict(self, obs, **kwargs):
        self.restore_env_with_obs(obs)
        _, chosen_action = self.expectiminimax(
            self.max_depth, self.env.agent_player, None, -float('inf'), float('inf'))
        return chosen_action, None
