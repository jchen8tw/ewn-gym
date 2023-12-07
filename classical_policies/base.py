from abc import abstractmethod
from typing import Tuple, Dict
import numpy as np


class PolicyBase:
    @abstractmethod
    def predict(self, obs: Dict[str, np.ndarray],
                **kwargs) -> Tuple[np.ndarray, None]:
        raise NotImplementedError
