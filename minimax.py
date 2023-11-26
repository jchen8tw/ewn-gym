from envs.ewn import EinsteinWuerfeltNichtEnv, Player

from tqdm import tqdm

class ExpectiminimaxAgent:
    def __init__(self, max_depth):
        self.max_depth = max_depth

    def expectiminimax(self, env: EinsteinWuerfeltNichtEnv, depth: int, player: Player, parent: Player | str, alpha: int, beta: int):
        if env.check_win() or depth == 0:
            return env.evaluate(), None

        if player == env.agent_player:  # Maximizing player
            best_val = -float('inf')
            best_action = None
            legal_actions = env.get_legal_actions(player)
            curr_dice_roll = env.dice_roll
            for action in legal_actions:
                env.set_dice_roll(curr_dice_roll)
                env.make_simulated_action(player, action)
                val, _ = self.expectiminimax(env, depth - 1, 'chance', player, alpha, beta)
                env.undo_simulated_action()
                if val > best_val:
                    best_val = val
                    best_action = action
                alpha = max(alpha, best_val)
                if beta <= alpha:
                    break
            return best_val, best_action

        elif player == env.get_opponent(env.agent_player):  # Minimizing player
            worst_val = float('inf')
            worst_action = None
            legal_actions = env.get_legal_actions(player)
            curr_dice_roll = env.dice_roll
            for action in legal_actions:
                env.set_dice_roll(curr_dice_roll)
                env.make_simulated_action(player, action)
                val, _ = self.expectiminimax(env, depth - 1, 'chance', player, alpha, beta)
                env.undo_simulated_action()
                if val < worst_val:
                    worst_val = val
                    worst_move = action
                beta = min(beta, worst_val)
                if beta <= alpha:
                    break
            return worst_val, worst_action

        elif player == 'chance':  # Chance node
            expected_val = 0
            next_player = env.agent_player if parent == env.get_opponent(env.agent_player) else env.get_opponent(env.agent_player)
            #print(f'chance node, next_player: {next_player}')
            for dice_roll in range(1, 7):
                env.set_dice_roll(dice_roll)
                #print(f'set dice roll to {dice_roll}')
                val, _ = self.expectiminimax(env, depth - 1, next_player, player, alpha, beta)
                expected_val += val / 6
            return expected_val, None

    def choose_action(self, env: EinsteinWuerfeltNichtEnv):
        _, chosen_action = self.expectiminimax(env, self.max_depth, env.agent_player, None, -float('inf'), float('inf'))
        return chosen_action


if __name__ == "__main__":
    
    num_simulations = 100
    win_count = 0

    for seed in tqdm(range(num_simulations)):
        # Testing the environment setup
        env = EinsteinWuerfeltNichtEnv(
            #render_mode="ansi",
            #render_mode="rgb_array",
            #render_mode="human",
            cube_layer=3,
            board_size=5,
            seed=seed
            )
        agent = ExpectiminimaxAgent(max_depth=3)
        obs, _  = env.reset()
        states = []

        step_count = 0

        while True:
            # env.render()
            states.append(env.render())
            #action = env.action_space.sample()
            action = agent.choose_action(env)
            env.set_dice_roll(obs['dice_roll'])
            obs, reward, done, trunc, info = env.step(action)
            if done:
                #print(info)
                if info['message'] == 'You won!':
                    win_count += 1
                if info['message'] != 'You won!' and info['message'] != 'You lost!':
                    print(info['message'])
                #print(info['message'])
                break
            
            step_count += 1

    print(f'win rate: {win_count / num_simulations * 100:.2f}%')

    """
    images = [Image.fromarray(state) for state in states]
    images = iter(images)
    image = next(images)
    image.save(
        f"ewn.gif",
        format="GIF",
        save_all=True,
        append_images=images,
        loop=0,
        duration=700,
    )
    """
