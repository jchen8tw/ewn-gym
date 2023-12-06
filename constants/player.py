from enum import Enum


class Player(Enum):
    TOP_LEFT = 1
    BOTTOM_RIGHT = 2
    # only used for minimax
    CHANCE = 3

    @classmethod
    def get_opponent(cls, player):
        if player == Player.TOP_LEFT:
            return Player.BOTTOM_RIGHT
        elif player == Player.BOTTOM_RIGHT:
            return Player.TOP_LEFT
        else:
            raise ValueError("Invalid player")
