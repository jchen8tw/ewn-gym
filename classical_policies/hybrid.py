
from .base import PolicyBase
import gymnasium as gym
from typing import Literal, Dict, List
import numpy as np
from constants import ClassicalPolicy


class HybridAgent(PolicyBase):
    """A hybrid policy that combines multiple policies together.
    """

    def __init__(self,
                 policy: List[ClassicalPolicy],
                 policy_prob: List[float],
                 original_env: gym.Env,
                 # whether to select action at each step or at the end of
                 # episode
                 selection_mode: Literal['step', 'episode'] = 'step',
                 # whether to use stochastic or ensemble mode to combine
                 hybrid_mode: Literal['stochastic', 'ensemble'] = 'stochastic',
                 cube_layer: int = 3,
                 board_size: int = 5,
                 **policy_kwargs):
        assert len(policy) == len(policy_prob), \
            'policy and policy_prob must have the same length'
        assert np.sum(policy_prob) == 1, \
            'policy_prob must sum to 1'

        self.selection_mode = selection_mode
        self.policy_list: List[PolicyBase] = []
        self.policy_prob = policy_prob
        for p in policy:
            assert p in ClassicalPolicy, f'Unsupported classical policy {p}'
            if p == ClassicalPolicy.minimax:
                from classical_policies import ExpectiMinimaxAgent
                max_depth: int = policy_kwargs.get('max_depth', 3)
                self.policy_list.append(ExpectiMinimaxAgent(
                    max_depth, cube_layer=cube_layer, board_size=board_size))
            elif p == ClassicalPolicy.random:
                from classical_policies import RandomAgent
                self.policy_list.append(RandomAgent(original_env))
            else:
                raise NotImplementedError("Policy not implemented yet")

        if selection_mode == 'episode':
            self.policy_idx = np.random.choice(
                len(self.policy_list), p=self.policy_prob)

    def predict(self, obs: Dict[str, np.ndarray], **kwargs):
        if self.selection_mode == 'step':
            policy_idx = np.random.choice(
                len(self.policy_list), p=self.policy_prob)
            return self.policy_list[policy_idx].predict(obs)
        elif self.selection_mode == 'episode':
            return self.policy_list[self.policy_idx].predict(obs)

    def reset(self):
        """This method is use for selection_mode == 'episode' to choose for a new policy for each episode.
        """
        self.policy_idx = np.random.choice(
            len(self.policy_list), p=self.policy_prob)
