import gymnasium as gym
import numpy as np
from enum import Enum
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
from typing import Tuple
from PIL import Image

COLORS = [
    "goldenrod",
    "white",
    "black",
]


class Player(Enum):
    TOP_LEFT = 1
    BOTTOM_RIGHT = 2


class EinsteinWuerfeltNichtEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, seed: int = 9487):
        super(EinsteinWuerfeltNichtEnv, self).__init__()
        self.board = np.zeros((5, 5), dtype=int)
        self.dice_roll = 1
        self.current_player = Player.TOP_LEFT
        self.setup_game()
        # Define action and observation space
        self.action_space = gym.spaces.Discrete(3)  # 3 possible moves
        self.observation_space = gym.spaces.Dict({
            "board": gym.spaces.Box(low=-6, high=6, shape=(5, 5), dtype=np.int8),
            "dice_roll": gym.spaces.Discrete(6)  # Dice values 1-6
        })
        np.random.seed(seed)
        self.render_init("Einstein Wuerfelt Nicht")

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
        self.previous_cube_pos = (x, y)

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
            self.current_cube_pos = (x, y)
        else:
            self.previous_cube_pos = self.current_cube_pos = None

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

    def render_board(self):
        grid_colors = COLORS
        cmap = colors.ListedColormap(grid_colors)
        board_color = self.board.copy()
        board_color[board_color > 0] = 1
        board_color[board_color == 0] = 0
        board_color[board_color < 0] = 2
        self.ax.imshow(board_color, cmap=cmap, vmin=0, vmax=2)

    def render_init(self, title="Einstein Wuerfelt Nicht"):
        plt.close("all")

        self.fig, self.ax = plt.subplots(
            figsize=(self.board.shape[1] + 1, self.board.shape[0] + 1))
        self.render_board()
        self.ax.grid(which="major", axis="both",
                     linestyle="-", color="gray", linewidth=2)
        self.ax.set_xticks(np.arange(-0.5, self.board.shape[1], 1))
        self.ax.set_yticks(np.arange(-0.5, self.board.shape[0], 1))
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.tick_params(length=0)
        self.previous_cube_pos: Tuple[int, int] = None
        self.current_cube_pos: Tuple[int, int] = None

        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                if self.board[i, j] == 0:
                    self.ax.text(
                        j,
                        i,
                        "",
                        ha="center",
                        va="center",
                        color="k",
                    )

                elif self.board[i, j] > 0:
                    self.ax.text(
                        j,
                        i,
                        f"{self.board[i, j]}",
                        ha="center",
                        va="center",
                        color="k",
                    )

                else:
                    self.ax.text(
                        j,
                        i,
                        f"{self.board[i, j]}",
                        ha="center",
                        va="center",
                        color="w",
                    )

        if title is not None:
            plt.title(title)

        self.ax.set_xlabel("dice: ")

        plt.tight_layout()

    def visualize(self, filename=None):
        if filename is not None:
            plt.savefig(filename)
        else:
            plt.show()

    def set_text_color(self, pos: Tuple[int, int], cube: int):
        """Sets the color and text of a grid cell.

        Args:
            pos (Tuple[int, int]): The position (row, column) of the cell to update.
            cube (int): The cube number of the current cell.
        """
        x, y = pos
        if cube == 0:
            cell_color = COLORS[0]
            text = ""
            text_color = "k"  # black (won't be visible as text is empty)
        elif cube < 0:
            cell_color = COLORS[2]
            text = str(cube)
            text_color = "white"
        else:  # cube < 0
            cell_color = COLORS[1]
            text = str(cube)
            text_color = "black"

        # Find and update the rectangle patch
        found_patch = False
        for artist in self.ax.patches:
            artist_x, artist_y = artist.get_xy()
            if int(artist_x + 0.5) == y and int(artist_y + 0.5) == x:
                artist.set_color(cell_color)
                found_patch = True
                break

        if not found_patch:
            # If no patch found, create a new one
            rect = patches.Rectangle(
                (y - 0.5, x - 0.5), 1, 1, color=cell_color)
            self.ax.add_patch(rect)

        # Find and update the text object
        for artist in self.ax.get_children():
            if isinstance(artist, matplotlib.text.Text):
                artist_x, artist_y = artist.get_position()
                if int(artist_x) == y and int(artist_y) == x:
                    artist.set_text(text)  # Set the new text
                    artist.set_color(text_color)  # Set the new text color
                    break

    def rgb_render(
        self,
    ) -> np.ndarray | None:
        """Render the environment as RGB image

        Args:
            title (str | None, optional): Title. Defaults to None.
        """
        if self.previous_cube_pos is not None:
            self.set_text_color(self.previous_cube_pos, 0)
        if self.current_cube_pos is not None:
            self.set_text_color(
                self.current_cube_pose,
                self.board[self.current_cube_pos[0], self.current_cube_pos[1]])
        self.ax.set_xlabel(f"dice: {self.dice_roll}")
        if self._step_count == 0:
            plt.pause(1)
        else:
            plt.pause(0.25)

    def get_rgb(self) -> np.ndarray:
        if self.previous_cube_pos is not None:
            self.set_text_color(self.previous_cube_pos, 0)
        if self.current_cube_pos is not None:
            self.set_text_color(
                self.current_cube_pos,
                self.board[self.current_cube_pos[0], self.current_cube_pos[1]])
        self.ax.set_xlabel(f"dice: {self.dice_roll}")
        self.fig.canvas.draw()
        buf = self.fig.canvas.buffer_rgba()
        data = np.asarray(buf)
        return data

    # def render(self, mode='human'):
    #     # Render the environment to the screen
    #     print("dice:")
    #     print(self.dice_roll)
    #     print("board:")
    #     print(self.board)

    def close(self):
        # Any cleanup goes here
        pass


if __name__ == "__main__":
    # Testing the environment setup
    env = EinsteinWuerfeltNichtEnv()
    env.reset()
    states = []
    while True:
        action = np.random.choice(3)
        rgb = env.get_rgb()
        states.append(rgb.copy())
        obs, reward, done, info = env.step(action)
        if done:
            break

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
