import warnings
import gymnasium as gym
from gymnasium.envs.registration import register

import numpy as np
import torch as th
import wandb
import argparse
from typing import Dict, List

import stable_baselines3 as sb3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_util import make_vec_env

warnings.filterwarnings("ignore")

# Note that we use the environment from envs/training_ewn.py
register(
    id='EWN-v0',
    # entry_point='envs:EinsteinWuerfeltNichtEnv'
    entry_point='envs:MiniMaxHeuristicEnv'
)


# Set hyper params (configurations) for training
# my_config = {
#     "run_id": "example",

#     # "algorithm": PPO,
#     "algorithm": A2C,
#     "policy_network": "MultiInputPolicy",
#     "save_path": "models/5x5",

#     "epoch_num": 5,
#     "cube_layer": 3,
#     "board_size": 5,
#     # "timesteps_per_epoch": 100,
#     "timesteps_per_epoch": 200000,
#     # "timesteps_per_epoch": 20000,
#     "eval_episode_num": 20,
#     # "learning_rate": 0.0002051234174866298,
#     "learning_rate": 3e-4,
#     "batch_size": 8,
#     "n_steps": 1,
#     "opponent_policy": "minimax",
#     "policy_kwargs": dict(activation_fn=th.nn.Tanh,
#                           #   net_arch=[dict(pi=[128, 64, 64], vf=[128, 64, 64])]
#                           )
# }


# def train(env, model, config):

#     current_best = 0

#     for epoch in range(config["epoch_num"]):

#         # Train agent using SB3
#         # Uncomment to enable wandb logging
#         model.learn(
#             total_timesteps=config["timesteps_per_epoch"],
#             reset_num_timesteps=False,
#             # callback=WandbCallback(
#             #     gradient_save_freq=100,
#             #     verbose=2,
#             # ),
#         )

#         # Evaluation
#         print(config["run_id"])
#         print("Epoch: ", epoch)
#         avg_score = 0
#         reward = []
#         reward_list = np.zeros(config["eval_episode_num"])
#         for seed in range(config["eval_episode_num"]):
#             done = [False]

#             env.seed(seed)
#             obs, info = env.reset()

#             # Interact with env using old Gym API
#             while not done[0]:
#                 action, _state = model.predict(obs, deterministic=True)
#                 obs, reward, done, info = env.step(action)

#             avg_score += reward[0] / config["eval_episode_num"]
#             # append the last reward of first env in vec_env
#             episode = seed
#             reward_list[episode] = reward[0]

#         print("Avg_score:  ", avg_score)
#         print("Reward_list:  ", reward_list)
#         winrate: float = np.count_nonzero(
#             np.array(reward_list) > 0) / config["eval_episode_num"]
#         print("Win rate:  ", winrate)
#         print()
#         # wandb.log(
#         #     {"avg_highest": avg_highest,
#         #      "avg_score": avg_score}
#         # )

#         # Save best model with highest win rate
#         if current_best < winrate:
#             print("Saving Model")
#             current_best = winrate
#             save_path = config["save_path"]
#             model.save(f"{save_path}/{epoch}")

#         print("---------------")

def return_model(config: Dict[str, str | int],
                 env: SubprocVecEnv) -> sb3.A2C | sb3.PPO:
    assert config["algorithm"] in ["A2C", "PPO"]
    model = getattr(sb3, config["algorithm"])
    if config["algorithm"] != "A2C":
        model = model(
            "MlpPolicy",
            env,
            # verbose=1,
            batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
            policy_kwargs=dict(
                activation_fn=th.nn.Tanh),
            # tensorboard_log=my_config["run_id"]
        )
    else:
        model = model(
            "MultiInputPolicy",
            env,
            # verbose=1,
            n_steps=config["n_steps"],
            learning_rate=config["learning_rate"],
            policy_kwargs=dict(
                activation_fn=th.nn.Tanh),
            # tensorboard_log=my_config["run_id"]
        )
    return model


def train(config=None):
    with wandb.init(config=config):

        config = dict(wandb.config)

        def make_env():
            env = gym.make(
                'EWN-v0',
                cube_layer=config["cube_layer"],
                board_size=config["board_size"],
                opponent_policy=config["opponent_policy"],
                goal_reward=config["goal_reward"],
                illegal_move_reward=config["illegal_move_reward"])
            return env

        env = SubprocVecEnv([make_env] * 8)
        if config["checkpoint"] is not None:
            model = return_model(config, env)
            model = model.load(config["checkpoint"])
        else:
            model = return_model(config, env)

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
            reward_list = np.zeros(config["eval_episode_num"])
            for seed in range(config["eval_episode_num"]):
                done = [False]

                env.seed(seed)
                obs, info = env.reset()

                # Interact with env using old Gym API
                print(obs)
                while not done[0]:
                    action, _state = model.predict(obs, deterministic=True)
                    obs, reward, done, info = env.step(action)

                avg_score += reward[0] / config["eval_episode_num"]
                # append the last reward of first env in vec_env
                episode = seed
                reward_list[episode] = reward[0]

            print("Avg_score:  ", avg_score)
            print("Reward_list:  ", reward_list)
            winrate: float = np.count_nonzero(
                np.array(reward_list) > 0) / config["eval_episode_num"]
            print("Win rate:  ", winrate)
            print()
            # wandb.log(
            #     {"avg_highest": avg_highest,
            #      "avg_score": avg_score}
            # )

            # Save best model with highest win rate
            if current_best < winrate:
                print("Saving Model")
                current_best = winrate
                save_path = f"models/{wandb.run.id}"
                model.save(f"{save_path}/{epoch}")

            print("---------------")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Trainer for EWN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    sub_parsers = parser.add_subparsers(
        dest="algorithm",
        required=True,
        help="Algorithm to use for training")
    parser_a2c = sub_parsers.add_parser("A2C")
    parser_a2c.add_argument(
        "-n",
        "--n_steps",
        type=int,
        default=1,
        help="The number of steps to run for each environment per update (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)")
    parser_ppo = sub_parsers.add_parser("PPO")
    parser_ppo.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=8,
        help="Minibatch size")

    parser.add_argument("--board_size", type=int, default=5, help="Board size")
    parser.add_argument("--cube_layer", type=int, default=3, help="Cube layer")
    parser.add_argument("-op",
                        "--opponent_policy",
                        type=str,
                        default="random",
                        choices=[
                            "random",
                            "minimax"], help="The policy of the opponent")
    parser.add_argument(
        "-e",
        "--epoch_num",
        type=int,
        default=5,
        help="Number of epochs to train")
    parser.add_argument(
        "-t",
        "--timesteps_per_epoch",
        type=int,
        default=200000)
    parser.add_argument(
        "--eval_episode_num",
        type=int,
        default=20,
        help="Number of episodes to evaluate per epoch")
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate")
    parser.add_argument(
        "-r",
        "--run_id",
        type=str,
        default="5x5",
        help="Run ID, also used as save path under model/")
    parser.add_argument(
        "--num_envs",
        type=int,
        default=8,
        help="Number of environments to run in parallel")
    parser.add_argument(
        "--illegal_move_reward",
        type=float,
        default=-1.0,
        help="reward for the agent when it makes an illegal move")
    parser.add_argument(
        "--goal_reward",
        type=float,
        default=10.0,
        help="reward for the agent when it wins.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to load from")

    return parser.parse_args()


if __name__ == "__main__":

    # Create wandb session (Uncomment to enable wandb logging)
    # run = wandb.init(
    #     project="RL-HW3",
    #     config=my_config,
    #     # sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    #     # id=my_config["run_id"]
    # )
    args = parse_args()
    config = dict(args._get_kwargs())
    # env = DummyVecEnv([make_env])
    # env = SubprocVecEnv([make_env] * 8)
    # print(env.get_attr("illegal_move_reward"))

    # Create model from loaded config and train
    # Note: Set verbose to 0 if you don't want info messages
    # model = my_config["algorithm"](
    #     my_config["policy_network"],
    #     env,
    #     # verbose=1,
    #     # batch_size=my_config["batch_size"],
    #     n_steps=my_config["n_steps"],
    #     learning_rate=my_config["learning_rate"],
    #     policy_kwargs=my_config["policy_kwargs"],
    #     # tensorboard_log=my_config["run_id"]
    # )
    # model = A2C.load(f"{my_config['save_path']}/1", env=env)
    train(config)
