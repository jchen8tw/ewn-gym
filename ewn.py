import gymnasium as gym
import numpy as np
from enum import Enum


class Player(Enum):
    TOP_LEFT = 1
    BOTTOM_RIGHT = 2


class EinsteinWuerfeltNichtEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(EinsteinWuerfeltNichtEnv, self).__init__()
        self.board = np.zeros((5, 5), dtype=int)
        self.dice_roll = 1
        self.current_player = Player.TOP_LEFT
        self.setup_game()
        # Define action and observation space
        self.action_space = gym.spaces.Discrete(4)  # 4 possible moves
        self.observation_space = gym.spaces.Dict({
            "board": gym.spaces.Box(low=-6, high=6, shape=(5, 5), dtype=np.int8),
            "dice_roll": gym.spaces.Discrete(6)  # Dice values 1-6
        })

    def setup_game(self):
        # Setting up the initial positions of the cubes for both players
        self.board[0, :3] = [1, 2, 3]
        self.board[1, :2] = [4, 5]
        self.board[2, 0] = 6
        self.board[-1, -3:] = [-1, -2, -3]
        self.board[-2, -2:] = [-4, -5]
        self.board[-3, -1] = -6

    def step(self, action):
        # Check if the game is already over
        if self.check_win():
            return self.board, 0, True, {"message": "Game already ended"}

        # Determine the cube to move based on the dice roll
        cube_to_move = self.find_cube_to_move(
            self.current_player, self.dice_roll)

        # Execute the move
        if cube_to_move is not None:
            self.execute_move(cube_to_move, action)

        # Check for win condition
        done = self.check_win()
        reward = 1 if done else 0

        # Switch turn
        self.current_player = Player.BOTTOM_RIGHT if self.current_player == Player.TOP_LEFT else Player.TOP_LEFT

        self.dice_roll = np.random.randint(
            1, 7)  # Roll the dice for the next turn
        return {"board": self.board,
                "dice_roll": self.dice_roll}, reward, done, {}

    def find_cube_to_move(self, player, dice_roll):
        # Adjust dice roll for player's cube numbers (positive for TOP_LEFT,
        # negative for BOTTOM_RIGHT)
        dice_roll = dice_roll if player == Player.TOP_LEFT else -dice_roll

        # Find available cubes for the player
        player_cubes = np.sort([cube for row in self.board for cube in row if (
            player == Player.TOP_LEFT and cube > 0) or (player == Player.BOTTOM_RIGHT and cube < 0)])

        # Find the cube matching the dice roll, or the nearest one
        if dice_roll in player_cubes:
            return dice_roll
        else:
            # Find nearest cube number
            nearest_cube = min(
                player_cubes, key=lambda x: (abs(x - dice_roll), x))
            return nearest_cube

    def execute_move(self, cube, action):
        # Find cube's current position
        pos = np.argwhere(self.board == cube)[0]
        x, y = pos[0], pos[1]

        # Determine new position based on action
        if cube > 0:  # TOP_LEFT player
            if action == 0:
                y += 1   # move right
            elif action == 1:
                x += 1   # move down
            elif action == 2:
                x += 1
                y += 1  # move diagonal down-right
        else:  # BOTTOM_RIGHT player
            if action == 0:
                y -= 1   # move left
            elif action == 1:
                x -= 1   # move up
            elif action == 2:
                x -= 1
                y -= 1  # move diagonal up-left

        # Check if move is within the board
        if 0 <= x < 5 and 0 <= y < 5:
            # Perform move and capture if necessary
            self.board[pos[0], pos[1]] = 0  # Remove cube from current position
            self.board[x, y] = cube  # Place cube in new position

    def check_win(self):
        # Check if any player has reached the opposite corner
        if self.board[0, 0] < 0 or self.board[4, 4] > 0:
            return True

        # Check if any player has removed all opponent's cubes
        if not np.any(self.board > 0):  # No TOP_LEFT cubes
            return True
        if not np.any(self.board < 0):  # No BOTTOM_RIGHT cubes
            return True

        return False

    def reset(self):
        self.setup_game()
        self.current_player = Player.TOP_LEFT
        self.dice_roll = np.random.randint(
            1, 7)  # Perform an initial dice roll
        return {"board": self.board, "dice_roll": self.dice_roll}

    def render(self, mode='human'):
        # Render the environment to the screen
        print("dice:")
        print(self.dice_roll)
        print("board:")
        print(self.board)

    def seed(self, seed: int = 9487):
        np.random.seed(seed)

    def close(self):
        # Any cleanup goes here
        pass


# Testing the environment setup
env = EinsteinWuerfeltNichtEnv()
env.seed()
env.reset()
env.render()
env.step(1)
env.render()
