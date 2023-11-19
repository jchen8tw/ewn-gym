import gymnasium as gym
import numpy as np
from enum import Enum
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
from typing import Tuple, Optional
from PIL import Image
import pygame
from pygame import gfxdraw


VIEWPORT_W = 500
VIEWPORT_H = 500


class Player(Enum):
    TOP_LEFT = 1
    BOTTOM_RIGHT = 2


class EinsteinWuerfeltNichtEnv(gym.Env):
    """
    The environment of Einstein Wuerfelt Nicht
    board_size(int): the size of the board(board_size * board_size)
    cube_layer(int): the number of layers of the cube
    seed(int): the seed for the random number generator
    agent_player: the player of the agent to train
    """
    metadata = {'render_modes': ['human', 'rgb_array', 'ansi']}

    def __init__(self, board_size: int = 5,
                 cube_layer: int = 3, seed: int = 9487, reward: float = 1., agent_player: Player = Player.TOP_LEFT, render_mode: Optional[str] = None, opponent_policy=None):
        super(EinsteinWuerfeltNichtEnv, self).__init__()

        # make sure the cube layer is legal
        # assert board_size % 2 == 1
        assert cube_layer < board_size - 1

        self.board: np.ndarray = np.zeros((board_size, board_size), dtype=int)
        cube_num: int = cube_layer * (cube_layer + 1) // 2
        print("Board size: ", board_size)
        print("Cube num: ", cube_num)
        # cube_pos[0] is the position of cube 1, cube_pos[1] is the position of
        # cube 2, etc.
        # cube_pos[-1], cube_pos[-2], etc. are the positions of the cubes of the
        # other player
        self.cube_pos = np.ma.zeros(
            (cube_num * 2,), dtype=[("x", int), ("y", int)])
        self.dice_roll: int = 1
        self.setup_game(cube_layer=cube_layer)
        # Define action and observation space
        self.action_space = gym.spaces.MultiDiscrete(
            [2, 3])  # 2 for chosing the large dice or the small dice ,3 possible moves
        self.observation_space = gym.spaces.Dict({
            "board": gym.spaces.Box(low=-6, high=6, shape=(5, 5), dtype=np.int8),
            "dice_roll": gym.spaces.Discrete(6)  # Dice values 1-6
        })
        # start with the top left player
        self.current_player = Player.TOP_LEFT
        self.agent_player = agent_player
        self.reward = reward
        if opponent_policy is not None:
            self.load_opponent_policy(opponent_policy)
        np.random.seed(seed)
        self.render_mode = render_mode
        # self.render_init("Einstein Wuerfelt Nicht")

    def roll_dice(self):
        self.dice_roll = np.random.randint(1, self.cube_pos.shape[0] // 2 + 1)

    def setup_game(self, cube_layer: int = 3):
        # Setting up the initial positions of the cubes for both players
        cnt = 1
        for i in range(1, cube_layer + 1):
            for j in range(0, i):
                self.board[j, i - j - 1] = cnt
                self.cube_pos[cnt - 1] = (j, i - j - 1)
                self.board[self.board.shape[0] - 1 - j,
                           self.board.shape[1] - i + j] = -cnt
                self.cube_pos[-cnt] = (self.board.shape[0] -
                                       1 - j, self.board.shape[1] - i + j)
                cnt += 1
        self.roll_dice()

    def check_win(self):
        # Check if any player has reached the opposite corner
        if self.board[0, 0] < 0 or self.board[-1, -1] > 0:
            return True

        # Check if any player has removed all opponent's cubes
        if not np.any(self.board > 0):  # No TOP_LEFT cubes
            return True
        if not np.any(self.board < 0):  # No BOTTOM_RIGHT cubes
            return True

        return False

    def find_cube_to_move(self, chose_larger: bool) -> int | None:
        # Adjust dice roll for player's cube numbers (positive for TOP_LEFT,
        # negative for BOTTOM_RIGHT)
        cube_pos_index: int = self.dice_roll - \
            1 if self.current_player == Player.TOP_LEFT else -self.dice_roll

        # Check if there is a cube to move
        if self.cube_pos.mask[cube_pos_index][0] == False:
            return cube_pos_index
        else:
            near_cube_pos_index = None
            if chose_larger:

                # Check if there is a larger cube to move
                if self.current_player == Player.TOP_LEFT:
                    for i in range(cube_pos_index + 1,
                                   self.cube_pos.shape[0] // 2):
                        if self.cube_pos.mask[i][0] == False:
                            near_cube_pos_index = i
                            break
                else:
                    for i in range(cube_pos_index - 1, -
                                   (self.cube_pos.shape[0] // 2 + 1), -1):
                        if self.cube_pos.mask[i][0] == False:
                            near_cube_pos_index = i
                            break
            else:
                # Check if there is a smaller cube to move
                if self.current_player == Player.TOP_LEFT:
                    for i in range(cube_pos_index - 1, -1, -1):
                        if self.cube_pos.mask[i][0] == False:
                            near_cube_pos_index = i
                            break
                else:
                    for i in range(cube_pos_index + 1, 0):
                        if self.cube_pos.mask[i][0] == False:
                            near_cube_pos_index = i
                            break

            return near_cube_pos_index

    def execute_move(self, cube_index: int, action: np.ndarray):
        # Find cube's current position
        pos = self.cube_pos[cube_index]
        x, y = pos[0], pos[1]

        cube = self.board[x, y]  # Get cube number

        assert cube != 0  # should be a cube not an empty cell

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
        if 0 <= x < self.board.shape[0] and 0 <= y < self.board.shape[1]:
            # Perform move and capture if necessary
            self.board[pos[0], pos[1]] = 0  # Remove cube from current position
            if self.board[x, y] != 0:
                remove_cube_index = self.board[x, y] - \
                    1 if self.board[x, y] > 0 else self.board[x, y]
                # Remove cube from cube_pos
                self.cube_pos[remove_cube_index] = np.ma.masked
            self.board[x, y] = cube  # Place cube in new position
            self.cube_pos[cube_index] = (x, y)  # Update cube_pos

    def load_opponent_policy(self, opponent_policy):
        """Load the opponent policy"""
        pass

    def opponent_action(self) -> np.ndarray:
        # currently the opponent is a random agent
        return self.action_space.sample()

    def switch_player(self):
        self.current_player = Player.BOTTOM_RIGHT if self.current_player == Player.TOP_LEFT else Player.TOP_LEFT

    def step(self, action: np.ndarray):

        # Determine the cube to move based on the dice roll
        cube_to_move_index = self.find_cube_to_move(action[0] == 1)

        # Execute the move
        if cube_to_move_index is not None:
            self.execute_move(cube_to_move_index, action[1])

        # Check for win condition
        if self.check_win():
            return self.board, self.reward, True, {
                "message": "You won!"}

        # Switch turn
        self.switch_player()
        self.roll_dice()  # Roll the dice for opponent's turn

        # Perform opponent's action
        opponent_action: np.ndarray = self.opponent_action()

        # Determine the cube to move based on the dice roll
        cube_to_move_index = self.find_cube_to_move(opponent_action[0] == 1)

        # Execute the move
        if cube_to_move_index is not None:
            self.execute_move(cube_to_move_index, opponent_action[1])
        if self.check_win():
            return self.board, -self.reward, True, {
                "message": "You lost!"}

        # Switch turn
        self.switch_player()
        # Roll the dice for the next turn
        self.roll_dice()

        return {"board": self.board,
                "dice_roll": self.dice_roll}, 0, False, {}

    def reset(self):
        self.setup_game()
        self.current_player = Player.TOP_LEFT
        self.roll_dice()  # Perform an initial dice roll
        return {"board": self.board, "dice_roll": self.dice_roll}

    def render(self):
        # Render the environment to the screen
        if self.render_mode == "ansi":
            print("dice:")
            print(self.dice_roll)
            print("board:")
            print(self.board)

    def close(self):
        # Any cleanup goes here
        pass


if __name__ == "__main__":
    # Testing the environment setup
    env = EinsteinWuerfeltNichtEnv(
        render_mode="ansi",
        cube_layer=3,
        board_size=5)
    # env.reset()
    states = []
    while True:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            break

    # images = [Image.fromarray(state) for state in states]
    # images = iter(images)
    # image = next(images)
    # image.save(
    #     f"ewn.gif",
    #     format="GIF",
    #     save_all=True,
    #     append_images=images,
    #     loop=0,
    #     duration=700,
    # )
