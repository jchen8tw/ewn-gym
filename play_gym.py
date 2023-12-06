import gymnasium as gym
from gymnasium.utils.play import play
import warnings
from gymnasium.envs.registration import register
import numpy as np
import pygame
from constants import ClassicalPolicy

warnings.filterwarnings("ignore")
register(
    id='EWN-v0',
    # entry_point='envs:MiniMaxHeuristicEnv'
    entry_point='envs:EinsteinWuerfeltNichtEnv'
)


env = gym.make("EWN-v0", render_mode="human",
               cube_layer=3,
               board_size=5,
               opponent_policy=ClassicalPolicy.minimax)

# Key mappings
key_to_action = {
    # chose larger number for player 1
    pygame.K_q: np.array([1, 0]),  # go horizontal
    pygame.K_w: np.array([1, 1]),  # go vertical
    pygame.K_e: np.array([1, 2]),  # go diagonal
    # chose smaller number for player 1
    pygame.K_a: np.array([0, 0]),
    pygame.K_s: np.array([0, 1]),
    pygame.K_d: np.array([0, 2]),
    # Add more key mappings here
}


done = False
obs, info = env.reset()
env.render()


while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        elif event.type == pygame.KEYDOWN:
            if event.key in key_to_action:
                action = key_to_action[event.key]
                obs, reward, done, truncate, info = env.step(action)
                env.render()
                print("Reward: {}, Done: {}, Truncate: {}, Info: {}".format(
                    reward, done, truncate, info.get("message", "")))
