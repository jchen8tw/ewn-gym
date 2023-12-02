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
    # entry_point='envs:EinsteinWuerfeltNichtEnv'
    entry_point='envs:MiniMaxHeuristicEnv'
)

# Set hyper params (configurations) for training
my_config = {
    "run_id": "example",

    # "algorithm": PPO,
    "algorithm": A2C,
    "policy_network": "MultiInputPolicy",
    "save_path": "models/5x5A2C_s",

    "epoch_num": 5,
    "cube_layer": 3,
    "board_size": 5,
    # "timesteps_per_epoch": 100,
    "timesteps_per_epoch": 200000,
    # "timesteps_per_epoch": 20000,
    "eval_episode_num": 20,
    # "learning_rate": 0.0002051234174866298,
    "learning_rate": 3e-4,
    "batch_size": 8,
    "n_steps": 1,
    #"opponent_policy": "minimax",
    "opponent_policy": "random",
    "policy_kwargs": dict(activation_fn=th.nn.Tanh,
                          #   net_arch=[dict(pi=[128, 64, 64], vf=[128, 64, 64])]
                          )
}


def make_env():
    env = gym.make(
        'EWN-v0',
        cube_layer=my_config["cube_layer"],
        board_size=my_config["board_size"],
        opponent_policy=my_config["opponent_policy"])
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
        reward_list = np.zeros(config["eval_episode_num"])
        for seed in range(config["eval_episode_num"]):
            done = [False]

            obs, info = env.reset(seed=seed)

            # Interact with env using old Gym API
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

    #env = DummyVecEnv([make_env])
    env = SubprocVecEnv([make_env] * 8)
    # print(env.get_attr("illegal_move_reward"))

    # Create model from loaded config and train
    # Note: Set verbose to 0 if you don't want info messages
    model = my_config["algorithm"](
        my_config["policy_network"],
        env,
        # verbose=1,
        # batch_size=my_config["batch_size"],
        n_steps=my_config["n_steps"],
        learning_rate=my_config["learning_rate"],
        policy_kwargs=my_config["policy_kwargs"],
        # tensorboard_log=my_config["run_id"]
    )
    #model = A2C.load(f"{my_config['save_path']}/1", env=env)
    train(env, model, my_config)
