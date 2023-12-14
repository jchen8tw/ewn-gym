from envs import EinsteinWuerfeltNichtEnv, MinimaxEnv
from classical_policies import ExpectiMinimaxAgent
import numpy as np
from constants import ClassicalPolicy
from statsmodels.stats.proportion import proportion_confint
from tqdm import trange
import argparse
import gymnasium as gym
from gymnasium.envs.registration import register

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
        description='Evaluate minimax agent', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--max_depth', type=int, default=3,
                        help='Max search depth')
    parser.add_argument('--heuristic', type=str, default='hybrid',
                        help='Heuristic used in searhcing')
    parser.add_argument('--num', type=int, default=100,
                        help='Number of rollouts')
    parser.add_argument('--render_last', action='store_true',
                        help='Render last rollout', default=False)
    parser.add_argument('--cube_layer', type=int, default=3,
                        help='Number of cube layers')
    parser.add_argument('--board_size', type=int, default=5,
                        help='Board size')
    parser.add_argument('--significance_level', type=float, default=0.05,
                        help='Board size')
    parser.add_argument('--opponent_policy', type=ClassicalPolicy.from_string, default=ClassicalPolicy.random, choices=list(ClassicalPolicy),
                        help='Opponent policy')
    parser.add_argument('--model_folder', type=str, default='alpha_zero_models',
                        help='folder of model')
    parser.add_argument('--model_name', type=str, default='checkpoint_100.pth.tar',
                        help='name of model')
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    env = gym.make(
        'EWN-v0',
        cube_layer=args.cube_layer,
        board_size=args.board_size,
        opponent_policy=args.opponent_policy,
        # render_mode='human',
        max_depth=args.max_depth,
        model_name=args.model_name,
        model_folder=args.model_folder,
    )
    
    agent = ExpectiMinimaxAgent(
        max_depth=args.max_depth,
        cube_layer=args.cube_layer,
        board_size=args.board_size,
        heuristic=args.heuristic)

    eval_num = args.num
    score = evaluation(env, agent, args.render_last, eval_num)

    print("Avg_score:  ", np.mean(score))
    winrate: float = np.count_nonzero(score > 0) / eval_num
    print("Avg win rate:  ", winrate)
    # print("Avg_highest:", np.sum(highest) / eval_num)
    #calculate (1-alpha)% confidence interval with {win_count} successes in {num_simulations} trials
    print(f'The {1-args.significance_level} confidence interval: {proportion_confint(count=winrate, nobs=eval_num, alpha=args.significance_level)}')

    print(f"Counts: (Total of {eval_num} rollouts)")
