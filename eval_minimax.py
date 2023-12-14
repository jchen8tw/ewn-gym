from envs import EinsteinWuerfeltNichtEnv, MinimaxEnv
from classical_policies import ExpectiMinimaxAgent
import numpy as np
from tqdm import tqdm
from constants import ClassicalPolicy
from statsmodels.stats.proportion import proportion_confint

if __name__ == "__main__":

    num_simulations = 1000
    cube_layer = 3
    board_size = 5
    alpha = 0.05
    
    
    env = EinsteinWuerfeltNichtEnv(
        # render_mode="ansi",
        # render_mode="rgb_array",
        # render_mode="human",
        cube_layer=cube_layer,
        board_size=board_size,
        # opponent_policy="models/5x5/1"
    )

    # minimax_env = MinimaxEnv(cube_layer=cube_layer, board_size=board_size)
    agent = ExpectiMinimaxAgent(
        max_depth=3,
        cube_layer=cube_layer,
        board_size=board_size,
        heuristic='hybrid')

    win_count = 0
    for seed in tqdm(range(num_simulations)):
        obs, _ = env.reset(seed=seed)
        states = []

        while True:
            states.append(env.render())
            action, _state = agent.predict(obs)
            obs, reward, done, trunc, info = env.step(action)
            if done:
                if info['message'] == 'You won!':
                    win_count += 1
                # print(info['message'])
                break

    print(f'win rate: {win_count / num_simulations * 100:.2f}%')
    #calculate (1-alpha)% confidence interval with {win_count} successes in {num_simulations} trials
    print(f'The {1-alpha} confidence interval for the true win rate: {proportion_confint(count=win_count, nobs=num_simulations, alpha=alpha)}')
