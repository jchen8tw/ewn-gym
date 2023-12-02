from envs import EinsteinWuerfeltNichtEnv, UCTEnv
from classical_policies import UCTAgent
import numpy as np
from tqdm import tqdm


if __name__ == "__main__":

    num_simulations = 1#000
    cube_layer = 3
    board_size = 5

    env = EinsteinWuerfeltNichtEnv(
        # render_mode="ansi",
        # render_mode="rgb_array",
        # render_mode="human",
        cube_layer=cube_layer,
        board_size=board_size,
        # opponent_policy="models/5x5/1"
    )

    env_UCT = UCTEnv(cube_layer=cube_layer, board_size=board_size)
    agent = UCTAgent(max_depth=3, env=env_UCT)

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
