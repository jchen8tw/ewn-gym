from enum import Enum


class ClassicalPolicy(Enum):

    random = "random"
    minimax = "minimax"
    uct = "uct"
    alpha_zero = "alpha_zero"
    mcts = "mcts"

    def __str__(self):
        return self.value

    @staticmethod
    def from_string(s):
        try:
            return ClassicalPolicy[s]
        except KeyError:
            return s
