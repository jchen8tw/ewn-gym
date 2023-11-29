from envs.ewn import EinsteinWuerfeltNichtEnv, Player
from typing import Tuple, Optional
import numpy as np
from tqdm import tqdm

class MinimaxEnv(EinsteinWuerfeltNichtEnv):
    def __init__(self, board_size: int = 5,
                 cube_layer: int = 3, seed: int = 9487, reward: float = 1., agent_player: Player = Player.TOP_LEFT, render_mode: Optional[str] = None, opponent_policy: str = "random"):
        super().__init__(board_size=board_size,
                         cube_layer=cube_layer, seed=seed, reward=reward, agent_player=agent_player, render_mode=render_mode, opponent_policy=opponent_policy)

        self.cube_num: int = cube_layer * (cube_layer + 1) // 2
    
    def get_opponent(self, player: Player):
        return Player.BOTTOM_RIGHT if player == Player.TOP_LEFT else Player.TOP_LEFT

    def set_dice_roll(self, roll: int):
        self.dice_roll = roll

    def evaluate(self):
        # Check if the agent player has won or lost
        # Used when searching to the end game
        if self.agent_player == Player.TOP_LEFT:
            if self.board[-1, -1] > 0 or not np.any(self.board < 0):  # Agent player wins
                return 10
            if self.board[0, 0] < 0 or not np.any(self.board > 0):  # Agent player loses
                return -10

        elif self.agent_player == Player.BOTTOM_RIGHT:
            if self.board[0, 0] < 0 or not np.any(self.board > 0):  # Agent player wins
                return 10
            if self.board[-1, -1] > 0 or not np.any(self.board < 0):  # Agent player loses
                return -10
        
        # Check the min distance to the goal and the number of remaining cubes
        # to calculate the value of the board
        board_height, board_width = self.board.shape
        board_length = max(board_height, board_width)
        score = 0
        min_dist_positive, min_dist_negative = float('inf'), float('inf')
        remaining_cubes_positive, remaining_cubes_negative = 0, 0
        for i in range(board_height):
            for j in range(board_width):
                if self.board[i, j] > 0:  # Pieces of TOP_LEFT player
                    curr_dist = max((board_height - 1 - i), (board_width - 1 - j))
                    min_dist_positive = min(min_dist_positive, curr_dist)
                    remaining_cubes_positive += 1
                elif self.board[i, j] < 0:  # Pieces of BOTTOM_RIGHT player
                    curr_dist = max((board_height - 1 - i), (board_width - 1 - j))
                    min_dist_negative = min(min_dist_negative, curr_dist)
                    remaining_cubes_negative += 1
        
        score += (board_length - min_dist_positive) * (1 / remaining_cubes_positive)
        score -= (board_length - min_dist_negative) * (1 / remaining_cubes_negative)

        """
        # Bad evaluation function
        # Calculate the distance of each player's pieces to the opposite corner
        for i in range(board_height):
            for j in range(board_width):
                if self.board[i, j] > 0:  # Pieces of TOP_LEFT player
                    # Add distance from bottom-right corner
                    score += (board_height - 1 - i) + (board_width - 1 - j)
                elif self.board[i, j] < 0:  # Pieces of BOTTOM_RIGHT player
                    # Subtract distance from top-left corner
                    score -= i + j
        # Consider the impact of the number of pieces remaining
        top_left_pieces = np.count_nonzero(self.board > 0)
        bottom_right_pieces = np.count_nonzero(self.board < 0)
        piece_difference = top_left_pieces - bottom_right_pieces
        some_factor = 1  # This is a parameter that might need adjustment
        score += piece_difference * some_factor
        """
        
        # Return a positive score for TOP_LEFT player, and negative for BOTTOM_RIGHT
        return score if self.agent_player == Player.TOP_LEFT else -score


class ExpectiMinimaxAgent:
    def __init__(self, max_depth, env):
        self.max_depth = max_depth
        self.env = env
    """
    def expectiminimax(self, env: EinsteinWuerfeltNichtEnv, depth: int, player: Player, parent: Player | str, alpha: int, beta: int):
        if env.check_win() or depth == 0:
            return env.evaluate(), None

        if player == env.agent_player:  # Maximizing player
            best_val = -float('inf')
            best_action = None
            legal_actions = env.get_legal_actions(player)
            curr_dice_roll = env.dice_roll
            for action in legal_actions:
                env.set_dice_roll(curr_dice_roll)
                env.make_simulated_action(player, action)
                val, _ = self.expectiminimax(env, depth - 1, 'chance', player, alpha, beta)
                env.undo_simulated_action()
                if val > best_val:
                    best_val = val
                    best_action = action
                alpha = max(alpha, best_val)
                if beta <= alpha:
                    break
            return best_val, best_action

        elif player == env.get_opponent(env.agent_player):  # Minimizing player
            worst_val = float('inf')
            worst_action = None
            legal_actions = env.get_legal_actions(player)
            curr_dice_roll = env.dice_roll
            for action in legal_actions:
                env.set_dice_roll(curr_dice_roll)
                env.make_simulated_action(player, action)
                val, _ = self.expectiminimax(env, depth - 1, 'chance', player, alpha, beta)
                env.undo_simulated_action()
                if val < worst_val:
                    worst_val = val
                    worst_move = action
                beta = min(beta, worst_val)
                if beta <= alpha:
                    break
            return worst_val, worst_action

        elif player == 'chance':  # Chance node
            expected_val = 0
            next_player = env.agent_player if parent == env.get_opponent(env.agent_player) else env.get_opponent(env.agent_player)
            #print(f'chance node, next_player: {next_player}')
            for dice_roll in range(1, 7):
                env.set_dice_roll(dice_roll)
                #print(f'set dice roll to {dice_roll}')
                val, _ = self.expectiminimax(env, depth - 1, next_player, player, alpha, beta)
                expected_val += val / 6
            return expected_val, None

    def choose_action(self, env: EinsteinWuerfeltNichtEnv):
        _, chosen_action = self.expectiminimax(env, self.max_depth, env.agent_player, None, -float('inf'), float('inf'))
        return chosen_action
    """ 
    def expectiminimax(self, depth: int, player: Player, parent: Player | str, alpha: int, beta: int):
        if self.env.check_win() or depth == 0:
            return self.env.evaluate(), None

        if player == self.env.agent_player:  # Maximizing player
            best_val = -float('inf')
            best_action = None
            legal_actions = self.env.get_legal_actions(player)
            curr_dice_roll = self.env.dice_roll
            for action in legal_actions:
                self.env.set_dice_roll(curr_dice_roll)
                self.env.make_simulated_action(player, action)
                val, _ = self.expectiminimax(depth - 1, 'chance', player, alpha, beta)
                self.env.undo_simulated_action()
                if val > best_val:
                    best_val = val
                    best_action = action
                alpha = max(alpha, best_val)
                if beta <= alpha:
                    break
            return best_val, best_action

        elif player == self.env.get_opponent(self.env.agent_player):  # Minimizing player
            worst_val = float('inf')
            worst_action = None
            legal_actions = self.env.get_legal_actions(player)
            curr_dice_roll = self.env.dice_roll
            for action in legal_actions:
                self.env.set_dice_roll(curr_dice_roll)
                self.env.make_simulated_action(player, action)
                val, _ = self.expectiminimax(depth - 1, 'chance', player, alpha, beta)
                self.env.undo_simulated_action()
                if val < worst_val:
                    worst_val = val
                    worst_move = action
                beta = min(beta, worst_val)
                if beta <= alpha:
                    break
            return worst_val, worst_action

        elif player == 'chance':  # Chance node
            expected_val = 0
            next_player = self.env.agent_player if parent == self.env.get_opponent(self.env.agent_player) else self.env.get_opponent(self.env.agent_player)
            #print(f'chance node, next_player: {next_player}')
            for dice_roll in range(1, 7):
                self.env.set_dice_roll(dice_roll)
                #print(f'set dice roll to {dice_roll}')
                val, _ = self.expectiminimax(depth - 1, next_player, player, alpha, beta)
                expected_val += val / 6
            return expected_val, None

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
    
    def predict(self, obs):
        self.restore_env_with_obs(obs)
        _, chosen_action = self.expectiminimax(self.max_depth, self.env.agent_player, None, -float('inf'), float('inf'))
        return chosen_action, None

if __name__ == "__main__":
    
    num_simulations = 1000
    cube_layer = 3
    board_size = 5

    env = EinsteinWuerfeltNichtEnv(
            #render_mode="ansi",
            #render_mode="rgb_array",
            #render_mode="human",
            cube_layer=cube_layer,
            board_size=board_size,
            )
    
    minimax_env = MinimaxEnv(cube_layer=cube_layer, board_size=board_size)
    agent = ExpectiMinimaxAgent(max_depth=3, env=minimax_env)

    win_count = 0
    for seed in tqdm(range(num_simulations)):
        obs, _  = env.reset(seed=seed)
        states = []
        
        while True:
            # env.render()
            states.append(env.render())
            #action = env.action_space.sample()
            action, _state = agent.predict(obs)
            obs, reward, done, trunc, info = env.step(action)
            if done:
                #print(info)
                if info['message'] == 'You won!':
                    win_count += 1
                if info['message'] != 'You won!' and info['message'] != 'You lost!':
                    print(info['message'])
                #print(info['message'])
                break

    print(f'win rate: {win_count / num_simulations * 100:.2f}%')

    """
    images = [Image.fromarray(state) for state in states]
    images = iter(images)
    image = next(images)
    image.save(
        f"ewn.gif",
        format="GIF",
        save_all=True,
        append_images=images,
        loop=0,
        duration=700,
    )
    """
