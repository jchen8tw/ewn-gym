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

register(
    id='EWN-eval-v0',
    entry_point='envs:EinsteinWuerfeltNichtEnv'
)


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
            seed=config["model_seed"]
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
            seed=config["model_seed"]
        )
    return model


def evaluate(config: Dict[str, any],
             model: sb3.A2C | sb3.PPO, current_best: float, epoch: int) -> float:
    # Evaluate agent using original Env
    avg_score = 0
    episode_num = config["eval_episode_num"]
    reward = []
    reward_list = np.zeros(episode_num)
    env = gym.make(
        'EWN-eval-v0',
        cube_layer=config["cube_layer"],
        board_size=config["board_size"],
        # evaluate with random opponent
        opponent_policy="random"
    )
    for seed in range(episode_num):
        done = False

        # Interact with env using old Gym API
        obs, info = env.reset(seed=seed)

        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, trunc, info = env.step(action)

        avg_score += reward / config["eval_episode_num"]
        # append the last reward of first env in vec_env
        episode = seed
        reward_list[episode] = reward

    print("Avg_score:  ", avg_score)
    print("Reward_list:  ", reward_list)
    winrate: float = np.count_nonzero(
        np.array(reward_list) > 0) / config["eval_episode_num"]
    print("Win rate:  ", winrate)
    print()
    wandb.log(
        {"win_rate": winrate,
         "avg_score": avg_score}
    )

    # Save best model with highest win rate
    if current_best < winrate:
        print("Saving Model")
        save_path = f"models/{wandb.run.id}"
        model.save(f"{save_path}/{epoch}")
        print("---------------")
        return winrate
    else:
        print("---------------")
        return current_best


def train(config=None):
    with wandb.init(project="ewn-gym", config=config):

        config = dict(wandb.config)

        def make_env():
            env = gym.make(
                'EWN-v0',
                cube_layer=config["cube_layer"],
                board_size=config["board_size"],
                opponent_policy=config["opponent_policy"],
                illegal_move_reward=config["illegal_move_reward"],
                illegal_move_tolerance=config["illegal_move_tolerance"],
                max_depth=config["max_depth"]
            )
            return env

        env = SubprocVecEnv([make_env] * config["num_envs"])
        env.seed(config["env_seed"])
        env.reset()
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
            print("Run id: ", wandb.run.id)
            print("Epoch: ", epoch)
            current_best = evaluate(config, model, current_best, epoch)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Trainer for EWN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    sub_parsers = parser.add_subparsers(
        dest="algorithm",
        required=True,
        help="Algorithm to use for training")
    parser_a2c = sub_parsers.add_parser("A2C", help="A2C algorithm")
    parser_a2c.add_argument(
        "-n",
        "--n_steps",
        type=int,
        default=1,
        help="The number of steps to run for each environment per update (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)")
    parser_ppo = sub_parsers.add_parser("PPO", help="PPO algorithm")
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
        "--max_depth",
        type=int,
        default=3,
        help="The max depth of minimax.")
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
        help="Number of episodes to evaluate per epoch, each episode num is also the seed for the environment.")
    parser.add_argument(
        "--model_seed",
        type=int,
        default=9487,
        help="Random seed for the sb3 model. This function is currently not supported."
    )
    parser.add_argument(
        "--env_seed",
        type=int,
        default=9487,
        help="Random seed for the environment."
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate")
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
        "--illegal_move_tolerance",
        type=int,
        default=10,
        help="Number of illegal moves the agent can make before it loses.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to load from")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = dict(args._get_kwargs())
    train(config)
