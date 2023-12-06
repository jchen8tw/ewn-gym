from .base import PolicyBase
import numpy as np
from typing import Tuple


class RandomAgent(PolicyBase):
    def __init__(self, env):
        from envs import EinsteinWuerfeltNichtEnv
        self.env: EinsteinWuerfeltNichtEnv = env

    def predict(self, obs, **kwargs) -> Tuple[np.ndarray, None]:
        legal_moves = self.env.get_legal_actions(self.env.current_player)
        action_idx = np.random.randint(0, len(legal_moves))
        action = np.array(legal_moves[action_idx])
        return action, None
