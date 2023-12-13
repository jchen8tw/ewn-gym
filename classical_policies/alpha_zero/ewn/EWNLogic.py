import numpy as np
from enum import Enum
from typing import Optional, List
from PIL import Image


VIEWPORT_SIZE = 700
FPS = 30
FONT_SIZE = VIEWPORT_SIZE // 25


class IllegalMove(Exception):
    pass


class Player(Enum):
    TOP_LEFT = 1
    BOTTOM_RIGHT = -1


class EinsteinWuerfeltNichtEnv():
    """
    The environment of Einstein Wuerfelt Nicht
    board_size(int): the size of the board(board_size * board_size)
    cube_layer(int): the number of layers of the cube
    seed(int): the seed for the random number generator
    agent_player: the player of the agent to train
    render_mode: the mode to render the environment
    opponent_policy: the policy of the opponent
    """
    metadata = {
        'render_modes': [
            'human',
            'rgb_array',
            'ansi'],
        'render_fps': FPS}

    def __init__(self, board_size: int = 5,
                 cube_layer: int = 3,
                 seed: int | None = None,
                 render_mode: Optional[str] = None,
                 ):

        # make sure the cube layer is legal
        # assert board_size % 2 == 1
        assert cube_layer < board_size - 1

        self.board: np.ndarray = np.zeros(
            (board_size, board_size), dtype=np.int16)
        self.cube_num: int = cube_layer * (cube_layer + 1) // 2
        self.cube_layer: int = cube_layer
        self.action_size: int = 6
        # cube_pos[0] is the position of cube 1, cube_pos[1] is the position of
        # cube 2, etc.
        # cube_pos[-1], cube_pos[-2], etc. are the positions of the cubes of the
        # other player
        self.cube_pos = np.ma.zeros(
            (self.cube_num * 2,), dtype=[("x", int), ("y", int)])
        self.dice_roll: int = 1

        # Setup the game
        self.reset(seed=seed)

        # Rendering
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.surf = None

    def roll_dice(self):
        self.dice_roll = np.random.randint(1, self.cube_num + 1)
        # self.dice_roll = self.observation_space["dice_roll"].sample()

    def setup_game(self):
        # Setting up the initial positions of the cubes for both players
        cnt = 1
        self.board.fill(0)
        self.cube_pos.fill(0)
        for i in range(1, self.cube_layer + 1):
            for j in range(0, i):
                self.board[j, i - j - 1] = cnt
                self.cube_pos[cnt - 1] = (j, i - j - 1)
                self.board[self.board.shape[0] - 1 - j,
                           self.board.shape[1] - i + j] = -cnt
                self.cube_pos[-cnt] = (self.board.shape[0] -
                                       1 - j, self.board.shape[1] - i + j)
                cnt += 1
        self.roll_dice()

    def set_board(self, board: np.ndarray, dice_roll: np.ndarray):
        # Set the board to the given board and dice roll
        new_board = np.zeros((board.shape[0], board.shape[1]), dtype=np.int16)
        for cube in range(self.cube_num):
            cur_board_layer = board[:, :, cube]
            cube_pos = np.nonzero(cur_board_layer)
            assert len(
                cube_pos[0]) <= 2, "at most 2 cube should be on the board for each cube number"
            top_left: bool = False
            bottom_right: bool = False
            for i in range(len(cube_pos[0])):
                if cur_board_layer[cube_pos[0][i], cube_pos[1][i]] == 1:
                    self.cube_pos[cube] = (cube_pos[0][i], cube_pos[1][i])
                    new_board[cube_pos[0][i], cube_pos[1][i]] = cube + 1
                    top_left = True
                elif cur_board_layer[cube_pos[0][i], cube_pos[1][i]] == -1:
                    self.cube_pos[-cube - 1] = (cube_pos[0][i], cube_pos[1][i])
                    new_board[cube_pos[0][i], cube_pos[1][i]] = -cube - 1
                    bottom_right = True
            # masking the cube_pos back if there are no cubes on the board
            if not top_left:
                self.cube_pos[cube] = np.ma.masked
            if not bottom_right:
                self.cube_pos[-cube - 1] = np.ma.masked
        self.board = new_board
        self.dice_roll = int(np.argmax(dice_roll)) + 1

    def check_win(self) -> int:
        # change to 1 0 -1 for player 1 win, not ended, player 2 win
        # Check if any player has reached the opposite corner
        if self.board[0, 0] < 0:
            return -1
        elif self.board[-1, -1] > 0:
            return 1

        # Check if any player has removed all opponent's cubes
        if not np.any(self.board > 0):  # No TOP_LEFT cubes
            return -1
        elif not np.any(self.board < 0):  # No BOTTOM_RIGHT cubes
            return 1

        return 0

    def find_near_cube(self, cube_pos_index: int,
                       chose_larger: bool, player: Player) -> int | None:
        # if player is None:
        #     player = self.current_player
        near_cube_pos_index: int | None = None
        if chose_larger:
            # Check if there is a larger cube to move
            if player == Player.TOP_LEFT:
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
            if player == Player.TOP_LEFT:
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

    def find_cube_to_move(self, chose_larger: bool,
                          player: Player) -> int:
        # if player is None:
        #     player = self.current_player

        # Adjust dice roll for player's cube numbers (positive for TOP_LEFT,
        # negative for BOTTOM_RIGHT)
        cube_pos_index: int = self.dice_roll - \
            1 if player == Player.TOP_LEFT else -self.dice_roll

        # Check if there is a cube to move
        if self.cube_pos.mask[cube_pos_index][0] == False:
            return cube_pos_index
        else:
            # Check if there is a cube to move
            near_cube_pos_index = self.find_near_cube(
                cube_pos_index, chose_larger, player)
            if near_cube_pos_index is not None:
                return near_cube_pos_index
            else:
                # ignore the chose_larger flag if there is no cube to move
                near_cube_pos_index = self.find_near_cube(
                    cube_pos_index, not chose_larger, player)
                assert near_cube_pos_index is not None
                return near_cube_pos_index

    def execute_move(self, cube_index: int, action: np.ndarray) -> bool:
        """ Execute the move of the cube and return whether the move is valid
        """
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
            return True
        else:
            return False

    def update_position(self, x: int, y: int, direction: int, cube: int):
        # Determine new position based on action and cube
        if cube > 0:
            if direction == 0:
                y += 1
            elif direction == 1:
                x += 1
            elif direction == 2:
                x += 1
                y += 1
        else:
            if direction == 0:
                y -= 1
            elif direction == 1:
                x -= 1
            elif direction == 2:
                x -= 1
                y -= 1
        return x, y

    def is_within_board(self, x: int, y: int):
        # Check if the position is within the board boundaries
        return 0 <= x < self.board.shape[0] and 0 <= y < self.board.shape[1]

    def get_cube_legal_directions(self, cube_pos_index: int) -> list[int]:
        cube_legal_directions = []
        pos = self.cube_pos[cube_pos_index]
        cube = self.board[pos[0], pos[1]]
        assert cube != 0  # should be a cube not an empty cell

        for direction in range(3):
            x, y = self.update_position(pos[0], pos[1], direction, cube)
            if self.is_within_board(x, y):
                cube_legal_directions.append(direction)

        return cube_legal_directions

    def get_legal_actions(self, player: Player) -> np.ndarray:
        # Legal move will be an array of actionsize
        # each element is either 1 or 0 representing whether the action is legal
        # the element represents the action in the following order respectively
        # [0,0] [0,1] [0,2] [1,0]... etc
        legal_actions = np.zeros(self.action_size, dtype=np.int8)

        # Adjust dice roll for player's cube numbers (positive for TOP_LEFT,
        # negative for BOTTOM_RIGHT)
        cube_pos_index: int = self.dice_roll - \
            1 if player == Player.TOP_LEFT else -self.dice_roll

        # Check if there is a cube to move
        if self.cube_pos.mask[cube_pos_index][0] == False:
            cube_legal_directions = self.get_cube_legal_directions(
                cube_pos_index)
            # cube_legal_actions = [[0, direction]
            #                       for direction in cube_legal_directions]
            # legal_actions.extend(cube_legal_actions)
            for direction in cube_legal_directions:
                legal_actions[direction] = 1  # [0, direction] = 1
            return legal_actions
        else:
            # Check if there is a larger cube to move
            near_cube_pos_index = self.find_near_cube(
                cube_pos_index, True, player)
            if near_cube_pos_index is not None:
                cube_legal_directions = self.get_cube_legal_directions(
                    near_cube_pos_index)
                # cube_legal_actions = [[1, direction]
                #                       for direction in cube_legal_directions]
                # legal_actions.extend(cube_legal_actions)
                for direction in cube_legal_directions:
                    legal_actions[direction + 3] = 1

            # Check if there is a smaller cube to move
            near_cube_pos_index = self.find_near_cube(
                cube_pos_index, False, player)
            if near_cube_pos_index is not None:
                cube_legal_directions = self.get_cube_legal_directions(
                    near_cube_pos_index)
                # cube_legal_actions = [[0, direction]
                #                       for direction in cube_legal_directions]
                # legal_actions.extend(cube_legal_actions)
                for direction in cube_legal_directions:
                    legal_actions[direction] = 1

            return legal_actions

    def step(self, action: int, player: Player):

        # action is now a 1d array of action size representing the action in the following order respectively
        # [0,0] [0,1] [0,2] [1,0]... etc

        action: np.ndarray = np.array([action // 3, action % 3])

        # Determine the cube to move based on the dice roll
        cube_to_move_index = self.find_cube_to_move(
            action[0] == 1, player=player)

        # Execute the move
        valid: bool = self.execute_move(cube_to_move_index, action[1])

        # Check if the move is valid
        if not valid:
            raise IllegalMove(
                f"Invalid move for {player.name}! End the game.")

        self.roll_dice()  # Roll the dice for opponent's turn

        # Check for win condition
        return {"board": self.board,
                "dice_roll": self.dice_roll}, self.check_win()

    def reset(self, seed: Optional[int] = None):
        np.random.seed(seed)
        self.setup_game()
        return {"board": self.board,
                "dice_roll": self.dice_roll}

    def render(self):
        # Render the environment to the screen
        if self.render_mode == "ansi":
            print("dice:")
            print(self.dice_roll)
            print("board:")
            print(self.board)
        elif self.render_mode == "human" or self.render_mode == "rgb_array":
            try:
                import pygame
                from pygame import gfxdraw
            except ImportError as e:
                raise Exception(
                    "pygame is not installed, run `pip install gymnasium[box2d]`"
                ) from e

            pygame.init()
            if self.render_mode == "human" and self.screen is None:
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (VIEWPORT_SIZE, VIEWPORT_SIZE + FONT_SIZE))
                pygame.display.set_caption("Einstein Wuerfelt Nicht")
            if self.clock is None:
                self.clock = pygame.time.Clock()

            self.surf = pygame.Surface(
                (VIEWPORT_SIZE, VIEWPORT_SIZE + FONT_SIZE))
            # draw the board
            self.surf.fill((211, 179, 104))
            font = pygame.font.SysFont("Arial", FONT_SIZE)
            dice_num = font.render(
                f"dice: {str(self.dice_roll)}", True, (0, 0, 0))
            self.surf.blit(dice_num, (0, VIEWPORT_SIZE))

            line_width = VIEWPORT_SIZE // self.board.shape[0]

            for i in range(1, self.board.shape[0]):
                gfxdraw.hline(self.surf, 0, VIEWPORT_SIZE,
                              i * line_width, (0, 0, 0))
                gfxdraw.vline(self.surf, i *
                              line_width, 0, VIEWPORT_SIZE, (0, 0, 0))
            # last line at the bottom
            gfxdraw.hline(self.surf, 0, VIEWPORT_SIZE,
                          VIEWPORT_SIZE, (0, 0, 0))
            # draw the cubes
            for i in range(self.cube_pos.shape[0]):
                if self.cube_pos.mask[i][0] != True:
                    x, y = self.cube_pos[i]
                    cube = self.board[x, y]
                    if cube > 0:
                        color = (255, 255, 255)
                        text_color = (0, 0, 0)
                    else:
                        color = (0, 0, 0)
                        text_color = (255, 255, 255)
                    gfxdraw.aacircle(self.surf, int((y + 0.5) * line_width), int(
                        (x + 0.5) * line_width), int(line_width * 0.4), color)
                    gfxdraw.filled_circle(self.surf, int((y + 0.5) * line_width), int(
                        (x + 0.5) * line_width), int(line_width * 0.4), color)
                    cube_num = font.render(
                        str(abs(cube)), True, text_color)
                    self.surf.blit(cube_num, (int((y + 0.5) * line_width) - FONT_SIZE // 3, int(
                        (x + 0.5) * line_width) - FONT_SIZE // 2))

            if self.render_mode == "human":
                assert self.screen is not None
                self.screen.blit(self.surf, (0, 0))
                pygame.event.pump()
                self.clock.tick(self.metadata["render_fps"])
                pygame.display.flip()
            elif self.render_mode == "rgb_array":
                return np.transpose(
                    np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)
                )

    def close(self):
        # Any cleanup goes here
        if self.screen is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()


if __name__ == "__main__":
    # Testing the environment setup
    env = EinsteinWuerfeltNichtEnv(
        render_mode="rgb_array",
        # render_mode="human",
        cube_layer=3,
        board_size=5)
    obs = env.reset()
    states = []
    done = False
    while True:
        # env.render()
        for player in [Player.TOP_LEFT, Player.BOTTOM_RIGHT]:
            states.append(env.render())
            action = np.random.choice(np.nonzero(
                env.get_legal_actions(player))[0])
            obs, done = env.step(action, player)
            if done:
                break
        if done:
            break
        # env.reset()

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
