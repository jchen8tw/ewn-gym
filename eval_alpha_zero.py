import gymnasium as gym
from gymnasium.envs.registration import register
from classical_policies import AlphaZeroAgent
# import plotly.graph_objects as go
# import plotly.express as px
import pandas as pd
from tqdm import trange
import argparse
from constants import ClassicalPolicy

import numpy as np
from collections import Counter
from statsmodels.stats.proportion import proportion_confint

register(
    id='EWN-v0',
    entry_point='envs:EinsteinWuerfeltNichtEnv'
)


def evaluation(env, model, render_last, eval_num=100) -> np.ndarray:
    score = np.zeros(eval_num)

    # Run eval_num times rollouts
    for seed in trange(eval_num):
        done = False
        # Set seed and reset env using Gymnasium API
        obs, info = env.reset(seed=seed)
        reward = 0

        while not done:
            # Interact with env using Gymnasium API
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)

        # Render the last board state of each episode
        # print("Last board state:")
        # env.render()
        # The episode number is same as the seed
        episode = seed
        score[episode] = reward

    # Render last rollout
    if render_last:
        print("Rendering last rollout")
        done = False
        obs, info = env.reset(seed=eval_num - 1)
        env.render()

        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            env.render()

    return score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Evaluate a trained model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_folder', type=str, default='alpha_zero_models',
                        help='folder of model')
    parser.add_argument('--model_name', type=str, default='checkpoint_242.pth.tar',
                        help='name of model')
    parser.add_argument('--num', type=int, default=1000,
                        help='Number of rollouts')
    parser.add_argument('--render_last', action='store_true',
                        help='Render last rollout', default=False)
    parser.add_argument('--cube_layer', type=int, default=3,
                        help='Number of cube layers')
    parser.add_argument('--board_size', type=int, default=5,
                        help='Board size')
    parser.add_argument(
        '--opponent_policy',
        type=ClassicalPolicy.from_string,
        default=ClassicalPolicy.random,
        help='Opponent policy')
    parser.add_argument(
        '--max_depth',
        type=int,
        default=3,
        help='Max depth for minimax')
    parser.add_argument('--significance_level', type=float, default=0.05,
                        help='Board size')
    parser.add_argument('--num_simulations', type=int, default=10,
                        help='Number of simulations')
    return parser.parse_args()


if __name__ == "__main__":
    # Change path name to load different models

    args = parse_args()

    env = gym.make(
        'EWN-v0',
        # render_mode='human',
        **args.__dict__
    )

    agent = AlphaZeroAgent(
        model_folder=args.model_folder,
        model_name=args.model_name,
        board_size=args.board_size,
        cube_layer=args.cube_layer,
        numMCTSSims=50,
        cpuct=1.0,
    )

    eval_num = args.num
    score = evaluation(env, agent, args.render_last, eval_num)

    print("Avg_score:  ", np.mean(score))
    winrate: float = np.count_nonzero(score > 0) / eval_num
    print("Avg win rate:  ", winrate)
    # print("Avg_highest:", np.sum(highest) / eval_num)
    # calculate (1-alpha)% confidence interval with {win_count} successes in
    # {num_simulations} trials
    print(f'The {1-args.significance_level} confidence interval: {proportion_confint(count=winrate, nobs=eval_num, alpha=args.significance_level)}')

    print(f"Counts: (Total of {eval_num} rollouts)")
