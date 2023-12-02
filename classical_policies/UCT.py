from constants import Player
from math import sqrt, log
import numpy as np
import random


class UCTAgent:
    def __init__(self, max_depth, env):
        self.max_depth = max_depth
        self.env = env

    def UCT(self, depth: int, player: Player,
                       parent: Player | str, alpha: int, beta: int):
        if self.env.check_win() or depth == 0:
            return self.env.evaluate(), None

        # selection
        while node.children:
            if node.explored_children < len(node.children):
                child = node.children[node.explored_children]
                node.explored_children += 1
                node = child
            else:
                node = max(node.children, key=ucb)
            _, reward, terminal, _ = state.step(node.action)
            sum_reward += reward
            actions.append(node.action)

        # expansion
        if not terminal:
            node.children = [Node(node, a) for a in combinations(state.action_space)]
            random.shuffle(node.children)

        # playout
        while not terminal:
            action = state.action_space.sample()
            _, reward, terminal, _ = state.step(action)
            sum_reward += reward
            actions.append(action)

            if len(actions) > self.max_depth:
                sum_reward -= 100
                break

        # remember best
        if best_reward < sum_reward:
            best_reward = sum_reward
            best_actions = actions

        # backpropagate
        while node:
            node.visits += 1
            node.value += sum_reward
            node = node.parent

    def restore_env_with_obs(self, obs):
        self.env.dice_roll = obs['dice_roll']
        self.env.board[:] = obs['board']

        for i in range(self.env.cube_num * 2):
            self.env.cube_pos[i] = np.ma.masked
        for i in range(self.env.board.shape[0]):
            for j in range(self.env.board.shape[1]):
                cube = self.env.board[i, j]
                if cube > 0:
                    self.env.cube_pos[cube - 1] = (i, j)
                elif cube < 0:
                    self.env.cube_pos[cube] = (i, j)

    def predict(self, obs, **kwargs):
        self.restore_env_with_obs(obs)
        _, chosen_action = self.UCT(
            self.max_depth, self.env.agent_player, None, -float('inf'), float('inf'))
        return chosen_action, None


def ucb(node):
    return node.value / node.visits + sqrt(log(node.parent.visits)/node.visits)


class Node:
    def __init__(self, parent, action):
        self.parent = parent
        self.action = action
        self.children = []
        self.explored_children = 0
        self.visits = 0
        self.value = 0
