from abc import abstractmethod
from typing import Tuple
import numpy as np


class PolicyBase:
    @abstractmethod
    def predict(self, obs, **kwargs) -> Tuple[np.ndarray, None]:
        raise NotImplementedError
