from ewn import EinsteinWuerfeltNichtEnv, Player

class ExpectiminimaxAgent:
    def __init__(self, max_depth):
        self.max_depth = max_depth

    def expectiminimax(self, env, depth, player, parent):
        if env.check_win() or depth == 0:
            return env.evaluate(), None

        if player == env.agent_player:
            best_val = -float('inf')
            best_move = None
            for move in env.get_legal_moves(player):
                env.make_simulated_move(move)
                val, _ = self.expectiminimax(env, depth - 1, 'chance', player)
                env.undo_simulated_move()
                if val > best_val:
                    best_val = val
                    best_move = move
            return best_val, best_move
        elif player == env.get_opponent(env.agent_player):
            worst_val = float('inf')
            worst_move = None
            for move in env.get_legal_moves(player):
                env.make_simulated_move(move)
                val, _ = self.expectiminimax(env, depth - 1, 'chance', player)
                env.undo_simulated_move()
                if val < worst_val:
                    worst_val = val
                    worst_move = move
            return worst_val, worst_move
        elif player == 'chance':
            expected_val = 0
            next_player = env.agent_player if parent == env.get_opponent(env.agent_player) else env.get_opponent(env.agent_player)
            for dice_roll in range(1, 7):
                env.set_dice_roll(dice_roll)
                val, _ = self.expectiminimax(env, depth - 1, next_player, player)
                expected_val += val / 6
            return expected_val, None

    def choose_move(self, env):
        _, chosen_move = self.expectiminimax(env, self.max_depth, env.agent_player, None)
        return chosen_move


if __name__ == "__main__":
    # Testing the environment setup
    env = EinsteinWuerfeltNichtEnv(
        #render_mode="ansi",
        #render_mode="rgb_array",
        render_mode="human",
        cube_layer=3,
        board_size=5)
    agent = ExpectiminimaxAgent(max_depth=3)
    obs = env.reset()
    states = []

    while True:
        # env.render()
        states.append(env.render())
        #action = env.action_space.sample()
        action = agent.choose_move(env)
        #env.dice_roll = env.original_dice_roll
        env.dice_roll = obs['dice_roll']
        obs, reward, done, info = env.step(action)
        if done:
            print(info)
            break

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
