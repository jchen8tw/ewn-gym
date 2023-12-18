import warnings
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium import spaces

import numpy as np
import torch as th
from torch import nn
import wandb
import argparse
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import stable_baselines3 as sb3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecVideoRecorder
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

from constants import ClassicalPolicy
from typing import Any

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


nn_args = dict({
    'lr': 0.001,
    'dropout': 0.3,
    'epochs': 10,
    'batch_size': 64,
    'cuda': th.cuda.is_available(),
    'num_channels': 512,
})

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=518)
        
        self.cube_num: int = config['cube_layer'] * (config['cube_layer'] + 1) // 2
        self.board_x, self.board_y, self.board_z = config['board_size'], config['board_size'], self.cube_num * 2
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.board_z, nn_args['num_channels'], 3, stride=1, padding=1),
            nn.BatchNorm2d(nn_args['num_channels']),
            nn.ReLU(),
            nn.Conv2d(nn_args['num_channels'], nn_args['num_channels'], 3, stride=1, padding=1),
            nn.BatchNorm2d(nn_args['num_channels']),
            nn.ReLU(),
            nn.Conv2d(nn_args['num_channels'], nn_args['num_channels'], 3, stride=1),
            nn.BatchNorm2d(nn_args['num_channels']),
            nn.ReLU(),
            nn.Conv2d(nn_args['num_channels'], nn_args['num_channels'], 3, stride=1),
            nn.BatchNorm2d(nn_args['num_channels']),
            nn.ReLU(),
            nn.Flatten(),
            )

    def forward(self, observations) -> th.Tensor:
        
        board = observations['board']
        dice_roll = observations['dice_roll']
        
        # Disard first value of dice roll due to start index 1
        dice_roll = dice_roll.view(dice_roll.size(0), -1)
        dice_roll = dice_roll[:, 1:]

        # One-hot encode the board
        # shape should be (batch_size, board, board, 12) after encoding
        device = board.device
        one_hot_shape = (*board.shape, 12)
        board_one_hot = th.zeros(one_hot_shape, dtype=th.float32).to(device)
        negative_positions = th.nonzero(board < 0)
        positive_positions = th.nonzero(board > 0)
        board_one_hot[negative_positions[:, 0], 
                negative_positions[:, 1], 
                negative_positions[:, 2], 
                -board[negative_positions[:, 0], negative_positions[:, 1], negative_positions[:, 2]].int() + 5] = 1.0
        board_one_hot[positive_positions[:, 0], 
                positive_positions[:, 1], 
                positive_positions[:, 2], 
                board[positive_positions[:, 0], positive_positions[:, 1], positive_positions[:, 2]].int() - 1] = 1.0
        
        out = board_one_hot.permute(0, 3, 1, 2)
        out = self.conv_layers(out)
        out = th.cat((out, dice_roll), dim=1)

        return out

class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 512,
        last_layer_dim_vf: int = 512,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        
        self.cube_num: int = config['cube_layer'] * (config['cube_layer'] + 1) // 2
        self.board_x, self.board_y, self.board_z = config['board_size'], config['board_size'], self.cube_num * 2
            
        self.fc_layers = nn.Sequential(
            nn.Linear(nn_args['num_channels'] * (self.board_x - 4) * (self.board_y - 4) + self.cube_num, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=nn_args['dropout']),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=nn_args['dropout'])
            )

        self.policy_net = self.fc_layers
        self.value_net = self.fc_layers
    
    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)
    
    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)
     

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)


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
            #"MultiInputPolicy",
            CustomActorCriticPolicy,
            env,
            # verbose=1,
            n_steps=config["n_steps"],
            learning_rate=config["learning_rate"],
            policy_kwargs=dict(
                #activation_fn=th.nn.Tanh,
                features_extractor_class=CustomCombinedExtractor,
                ),
            # tensorboard_log=my_config["run_id"]
            seed=config["model_seed"]
        )
        print(model.policy)
    return model


def evaluate(config: Dict[str, Any],
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
        opponent_policy=ClassicalPolicy.random,
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
    print("Reward_list(first 10):  ", reward_list[:10])
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
        #model.save(f"{save_path}/{epoch}")
        model.save(f"{save_path}/best")
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
                **config
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
                        type=ClassicalPolicy.from_string,
                        default=ClassicalPolicy.random,
                        choices=list(ClassicalPolicy), help="The policy of the opponent")
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
    parser.add_argument(
        "--alpha_model_name",
        type=str,
        default=None,
        dest="model_name",
        help="model name of the alpha zero model")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = dict(args._get_kwargs())
    print(config)
    train(config)
