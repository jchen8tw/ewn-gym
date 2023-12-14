from typing import Tuple
import numpy as np
from classical_policies.base import PolicyBase
import numpy as np
from .MCTS import MCTS
from .ewn.EWNGame import EWNGame
from .ewn.EWNPlayers import *
from .ewn.pytorch.NNet import NNetWrapper as NNet
from .utils import *
from typing import Dict


class AlphaZeroAgent(PolicyBase):
    def __init__(
            self,
            model_folder: str = "./alpha_zero_models",
            model_name: str = "checkpoint_40.pth.tar",
            board_size: int = 5,
            cube_layer: int = 3,
            numMCTSSims: int = 50,
            cpuct: float = 1.0,
            **kwargs):
        g = EWNGame(board_size=board_size, cube_layer=cube_layer)
        n1 = NNet(g)
        n1.load_checkpoint(model_folder, model_name)
        args1 = dotdict({'numMCTSSims': numMCTSSims, 'cpuct': cpuct})
        self.mcts = MCTS(g, n1, args1)
        self.index2action = {
            0: np.array([0, 0]),  # go horizontal
            1: np.array([0, 1]),  # go vertical
            2: np.array([0, 2]),  # go diagonal
            3: np.array([1, 0]),
            4: np.array([1, 1]),
            5: np.array([1, 2]),
        }
        self.cube_layer = cube_layer
        self.board_size = board_size

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

    def predict(self, obs, **kwargs) -> Tuple[np.ndarray, None]:
        """ predict action from obs """
        obs = self.ewn_obs_one_hot(obs)

        action = self.index2action[np.argmax(
            self.mcts.getActionProb(obs, temp=0))]
        return action, None
