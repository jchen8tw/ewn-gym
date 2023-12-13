from __future__ import print_function
import numpy as np
from .EWNLogic import EinsteinWuerfeltNichtEnv, Player
from .Game import Game
import sys
from typing import Dict
sys.path.append('..')


class EWNGame(Game):

    def __init__(self,
                 board_size: int = 5,
                 cube_layer: int = 3,
                 seed: int | None = None,
                 render_mode: str = 'human'):
        self.ewn = EinsteinWuerfeltNichtEnv(
            board_size=board_size,
            cube_layer=cube_layer,
            seed=seed,
            render_mode=render_mode)

    def ewn_obs_one_hot(self) -> Dict[str, np.ndarray]:
        board_size = self.ewn.board.shape[0]
        cube_num = self.ewn.cube_num  # cube num for each player

        board = np.zeros((board_size, board_size, cube_num), dtype=int)
        for i in range(cube_num):
            if not self.ewn.cube_pos.mask[i][0]:
                x, y = self.ewn.cube_pos[i]
                board[x, y, i] = 1
            if not self.ewn.cube_pos.mask[-i - 1][0]:
                x, y = self.ewn.cube_pos[-i - 1]
                board[x, y, i] = -1

        dice_roll = np.zeros(cube_num, dtype=int)
        dice_roll[self.ewn.dice_roll - 1] = 1
        return {"board": board, "dice_roll": dice_roll}

    @staticmethod
    def get_ewn_player(player: int) -> Player:
        if player == 1:
            return Player.TOP_LEFT
        elif player == -1:
            return Player.BOTTOM_RIGHT
        else:
            raise ValueError("player must be 1 or -1")

    def getInitBoard(self) -> Dict[str, np.ndarray]:
        # return a dict of initial board (3d numpy board, each layer are for a cube num) and dice_roll(numpy
        # array)
        self.ewn.reset()
        return self.ewn_obs_one_hot()

    def getBoardSize(self) -> tuple[int, int, int]:
        # (a,b) tuple
        return (self.ewn.board.shape[0],
                self.ewn.board.shape[1], self.ewn.cube_num)

    def getActionSize(self) -> int:
        # return number of actions
        return self.ewn.action_size

    def getNextState(self, board: Dict[str, np.ndarray], player: int,
                     action: int) -> tuple[Dict[str, np.ndarray], int]:
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        self.ewn.set_board(board["board"], board["dice_roll"])
        self.ewn.step(action=action, player=self.get_ewn_player(player))
        return (self.ewn_obs_one_hot(), -player)

    def getValidMoves(
            self, board: Dict[str, np.ndarray], player: int) -> np.ndarray:
        # return a fixed size binary vector
        self.ewn.set_board(board["board"], board["dice_roll"])
        valids = self.ewn.get_legal_actions(self.get_ewn_player(player))
        return valids

    def getGameEnded(self, board: Dict[str, np.ndarray], player: int) -> int:
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        self.ewn.set_board(board["board"], board["dice_roll"])
        return self.ewn.check_win() * player

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        if player == 1:
            return {"board": board["board"],
                    "dice_roll": board["dice_roll"]}
        elif player == -1:
            return {"board": -np.rot90(board["board"], 2, axes=(0, 1)),
                    "dice_roll": board["dice_roll"]}

    def getSymmetries(self, board, pi):
        # mirror, rotational
        # no need for EiNSTEIN WUERFELT NICHT since it has no rotation or
        # mirror
        assert (len(pi) == self.ewn.action_size)
        return [(board, pi)]

    def stringRepresentation(self, board):
        return str(board)

    def display(self, board):
        self.ewn.render()
