from constants import Player
import numpy as np
from .base import PolicyBase

from multiprocessing import Pool
import copy
import random


class MctsAgent(PolicyBase):
    def __init__(self, cube_layer: int, board_size: int):
        from envs import MinimaxEnv
        self.env = MinimaxEnv(
            cube_layer=cube_layer,
            board_size=board_size)

    def simulate(self, env_copy, num_simulations=10) -> int:
        """ simulate till game over and get win count """
        win_count = 0
        for _ in range(num_simulations):
            action_count = 0
            curr_player = Player.TOP_LEFT
            # Simulate till game over
            while not env_copy.check_win():
                curr_player = Player.get_opponent(curr_player)
                env_copy.set_dice_roll(random.randint(1, 6))
                curr_legal_actions = env_copy.get_legal_actions(curr_player)
                chosen_action_idx = random.randint(0, len(curr_legal_actions) - 1)
                curr_legal_action = curr_legal_actions[chosen_action_idx]
                env_copy.make_simulated_action(curr_player, curr_legal_action)
                action_count += 1
            # Check if the agent player has won
            if env_copy.board[-1, -
                          1] > 0 or not np.any(env_copy.board < 0):
                win_count += 1
            # Restore the env after simulation
            for _ in range(action_count):
                env_copy.undo_simulated_action()
        return win_count

    def tree_search_and_get_move_mp(self) -> np.ndarray:
        """ mcts with multiprocessing """
        legal_actions = self.env.get_legal_actions(Player.TOP_LEFT)
        
        num_copies_per_env = 5
        envs = []
        for legal_action in legal_actions:
            self.env.make_simulated_action(Player.TOP_LEFT, legal_action)
            for _ in range(num_copies_per_env):
                envs.append(copy.deepcopy(self.env)) 
            self.env.undo_simulated_action()

        pool = Pool()  # Create a pool of processes
        tmp_win_counts = pool.map(self.simulate, envs)
        win_counts = [sum(tmp_win_counts[i : i+num_copies_per_env]) for i in range(0, len(tmp_win_counts), num_copies_per_env)]
        #print(f'tmp_win_counts: {tmp_win_counts}')
        #print(f'win_counts: {win_counts}')

        pool.close()
        pool.join()
    
        best_action = legal_actions[np.argmax(win_counts)]
        return np.array(best_action)

    def tree_search_and_get_move(self) -> np.ndarray:
        """ mcts without multiprocessing """
        legal_actions = self.env.get_legal_actions(Player.TOP_LEFT)
        win_counts = []
    
        for legal_action in legal_actions:
            self.env.make_simulated_action(Player.TOP_LEFT, legal_action)
            env_copy = copy.deepcopy(self.env)
            win_count = self.simulate(env_copy)
            win_counts.append(win_count)
            self.env.undo_simulated_action()
    
        #print(f'win_counts: {win_counts}')
        best_action = legal_actions[np.argmax(win_counts)]
        return np.array(best_action)

    def restore_env_with_obs(self, obs):
        # Restore dice and board
        self.env.dice_roll = obs['dice_roll']
        self.env.board[:] = obs['board']

        # Restore the cube positions
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
        chosen_action = self.tree_search_and_get_move_mp()
        #chosen_action = self.tree_search_and_get_move()
        return chosen_action, None
