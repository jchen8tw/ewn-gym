import numpy as np
import pygame


class HumanEWNPlayer():
    def __init__(self, game):
        self.game = game
        self.key_to_action = {
            # chose larger number for player 1
            pygame.K_q: 3,  # go horizontal
            pygame.K_w: 4,  # go vertical
            pygame.K_e: 5,  # go diagonal
            # chose smaller number for player 1
            pygame.K_a: 0,
            pygame.K_s: 1,
            pygame.K_d: 2,
            # Add more key mappings here
        }

    def play(self, board):
        # display(board)
        valid = self.game.getValidMoves(board, 1)
        print("Available moves: ", end="")
        for i in range(len(valid)):
            if valid[i]:
                print(i, end=" ")
        print()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key in self.key_to_action:
                        action = self.key_to_action[event.key]
                        if valid[action]:
                            return action
                        else:
                            print('Invalid move')
