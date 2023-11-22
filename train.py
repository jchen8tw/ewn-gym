import warnings
import gymnasium as gym
from gymnasium.envs.registration import register

import numpy as np
import torch as th


from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env

warnings.filterwarnings("ignore")
register(
    id='EWN-v0',
    entry_point='envs:EinsteinWuerfeltNichtEnv'
)

# Set hyper params (configurations) for training
my_config = {
    "run_id": "example",

    "algorithm": PPO,
    "policy_network": "MultiInputPolicy",
    "save_path": "models/5x5PPO",

    "epoch_num": 5,
    "cube_layer": 3,
    "board_size": 5,
    # "timesteps_per_epoch": 100,
    # "timesteps_per_epoch": 100000,
    "timesteps_per_epoch": 20000,
    "eval_episode_num": 20,
    # "learning_rate": 0.0002051234174866298,
    "learning_rate": 3e-4,
    "batch_size": 8,
    # "n_steps": 1,
    "policy_kwargs": dict(activation_fn=th.nn.Tanh,
                          #   net_arch=[dict(pi=[128, 64, 64], vf=[128, 64, 64])]
                          )
}


def make_env():
    env = gym.make(
        'EWN-v0',
        cube_layer=my_config["cube_layer"],
        board_size=my_config["board_size"])
    return env


def train(env, model, config):

    current_best = 0

    for epoch in range(config["epoch_num"]):

        # Train agent using SB3
        # Uncomment to enable wandb logging
        model.learn(
            total_timesteps=config["timesteps_per_epoch"],
            reset_num_timesteps=False,
            # callback=WandbCallback(
            #     gradient_save_freq=100,
            #     verbose=2,
            # ),
        )

        # Evaluation
        print(config["run_id"])
        print("Epoch: ", epoch)
        avg_score = 0
        reward = []
        reward_list = []
        for seed in range(config["eval_episode_num"]):
            done = [False]

            # Set seed using old Gym API
            env.seed(seed)
            obs = env.reset()

            # Interact with env using old Gym API
            while not done[0]:
                action, _state = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)

            avg_score += reward[0] / config["eval_episode_num"]
            reward_list.append(reward[0])

        print("Avg_score:  ", avg_score)
        print("Reward_list:  ", reward_list)
        print("Win rate:  ", (avg_score + 1) / 2)
        print()
        # wandb.log(
        #     {"avg_highest": avg_highest,
        #      "avg_score": avg_score}
        # )

        # Save best model with highest win rate
        if current_best < (avg_score + 1) / 2:
            print("Saving Model")
            current_best = (avg_score + 1) / 2
            save_path = config["save_path"]
            model.save(f"{save_path}/{epoch}")

        print("---------------")


if __name__ == "__main__":

    # Create wandb session (Uncomment to enable wandb logging)
    # run = wandb.init(
    #     project="RL-HW3",
    #     config=my_config,
    #     # sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    #     # id=my_config["run_id"]
    # )

    # env = DummyVecEnv([make_env])
    env = SubprocVecEnv([make_env] * 8)
    # print(env.get_attr("illegal_move_reward"))

    # Create model from loaded config and train
    # Note: Set verbose to 0 if you don't want info messages
    model = my_config["algorithm"](
        my_config["policy_network"],
        env,
        # verbose=1,
        batch_size=my_config["batch_size"],
        # n_steps=my_config["n_steps"],
        learning_rate=my_config["learning_rate"],
        policy_kwargs=my_config["policy_kwargs"],
        # tensorboard_log=my_config["run_id"]
    )
    train(env, model, my_config)
