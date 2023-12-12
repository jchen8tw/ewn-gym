from constants import Player
import numpy as np
from .base import PolicyBase
from multiprocessing import Pool
import copy


class MctsAgent(PolicyBase):
    def __init__(self, cube_layer: int, board_size: int):
        from envs import MinimaxEnv
        self.env = MinimaxEnv(
            cube_layer=cube_layer,
            board_size=board_size)

    def simulate(self, env_copy, num_simulations=25):
        curr_player = env_copy.agent_player

        win_count = 0
        for _ in range(num_simulations):
            action_count = 0
            # Simulate till game over
            while not env_copy.check_win():
                curr_player = Player.get_opponent(curr_player)
                env_copy.set_dice_roll(np.random.randint(1, 7))
                curr_legal_actions = env_copy.get_legal_actions(curr_player)
                chosen_action_idx = np.random.randint(len(curr_legal_actions))
                curr_legal_action = curr_legal_actions[chosen_action_idx]
                env_copy.make_simulated_action(curr_player, curr_legal_action)
                action_count += 1
            # Check if the agent player has won or lost
            if env_copy.agent_player == Player.TOP_LEFT:
                if env_copy.board[-1, -
                              1] > 0 or not np.any(env_copy.board < 0):  # Agent player wins
                    win_count += 1
            elif env_copy.agent_player == Player.BOTTOM_RIGHT:
                if env_copy.board[0, 0] < 0 or not np.any(
                        env_copy.board > 0):  # Agent player wins
                    win_count += 1
            for _ in range(action_count):
                env_copy.undo_simulated_action()
        return win_count

    def tree_search_and_get_move_mp(self):
        legal_actions = self.env.get_legal_actions(self.env.agent_player)
        
        env_times = 4
        envs = []
        for legal_action in legal_actions:
            self.env.make_simulated_action(self.env.agent_player, legal_action)
            for _ in range(env_times):
                envs.append(copy.deepcopy(self.env)) 
            self.env.undo_simulated_action()
            
        pool = Pool()  # Create a pool of processes
        tmp_win_counts = pool.map(self.simulate, envs)
        win_counts = [sum(tmp_win_counts[i:i + env_times]) for i in range(0, len(tmp_win_counts), env_times)]

        pool.close()
        pool.join()
    
        #print(f'win_counts: {win_counts}')
        best_action = legal_actions[np.argmax(win_counts)]
        return best_action

    def tree_search_and_get_move(self):
        curr_player = self.env.agent_player
        legal_actions = self.env.get_legal_actions(curr_player)
        win_rates = np.zeros(len(legal_actions))
        simulations_per_node = 50
        #print(f'legal_actions: {legal_actions}')
    
        for i, legal_action in enumerate(legal_actions):
            #print(f'board: {self.env.board}')
            self.env.make_simulated_action(curr_player, legal_action)
            curr_win_count = 0
            for _ in range(simulations_per_node):
                action_count = 0
                # Simulate till game over
                while not self.env.check_win():
                    curr_player = Player.get_opponent(curr_player)
                    self.env.set_dice_roll(np.random.randint(1, 7))
                    #print(f'self.env.dice_roll: {self.env.dice_roll}')
                    curr_legal_actions = self.env.get_legal_actions(curr_player)
                    #print(f'curr_legal_actions: {curr_legal_actions}')
                    chosen_action_idx = np.random.randint(len(curr_legal_actions))
                    curr_legal_action = curr_legal_actions[chosen_action_idx]
                    self.env.make_simulated_action(curr_player, curr_legal_action)
                    action_count += 1
                # Check if the agent player has won or lost
                if self.env.agent_player == Player.TOP_LEFT:
                    if self.env.board[-1, -
                                  1] > 0 or not np.any(self.env.board < 0):  # Agent player wins
                        curr_win_count += 1
                elif self.env.agent_player == Player.BOTTOM_RIGHT:
                    if self.env.board[0, 0] < 0 or not np.any(
                            self.env.board > 0):  # Agent player wins
                        curr_win_count += 1
                for _ in range(action_count):
                    self.env.undo_simulated_action()
            self.env.undo_simulated_action()
            #print(f'curr_win_count: {curr_win_count}')
            win_rates[i] = curr_win_count
    
        win_rates = win_rates / simulations_per_node
        #print(f'win_rates: {win_rates}')
        best_action = legal_actions[np.argmax(win_rates)]
        return best_action

    def restore_env_with_obs(self, obs):
        self.env.dice_roll = obs['dice_roll']
        self.env.board[:] = obs['board']

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
        return chosen_action, None
