import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import PPO, A2C
# import plotly.graph_objects as go
# import plotly.express as px
import pandas as pd


import numpy as np
from collections import Counter

register(
    id='EWN-v0',
    entry_point='envs:EinsteinWuerfeltNichtEnv'
)


def evaluation(env, model, render_last, eval_num=100):
    score = []

    # Run eval_num times rollouts
    for seed in range(eval_num):
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

        score.append(reward)

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


if __name__ == "__main__":
    # Change path name to load different models
    model_path = "models/sample_model/2"
    env = gym.make('EWN-v0', render_mode="human")

    # Load model with SB3
    # Note: Model can be loaded with arbitrary algorithm class for evaluation
    # (You don't necessarily need to use PPO for training)
    model = A2C.load(model_path)

    eval_num = 1000
    score = evaluation(env, model, True, eval_num)

    print("Avg_score:  ", np.sum(score) / eval_num)
    print("Avg win rate:  ", (np.sum(score) / eval_num + 1) / 2)
    # print("Avg_highest:", np.sum(highest) / eval_num)

    print(f"Counts: (Total of {eval_num} rollouts)")
    # df = pd.DataFrame(highest, columns=["highest"]).astype(str)
    # fig = px.histogram(
    #     df,
    #     x="highest",
    #     category_orders=dict(
    #         highest=[
    #             "64",
    #             "128",
    #             "256",
    #             "512"]))
    # fig.show()
