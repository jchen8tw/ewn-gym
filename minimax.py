from envs.ewn import EinsteinWuerfeltNichtEnv, Player

from tqdm import tqdm

class ExpectiminimaxAgent:
    def __init__(self, max_depth):
        self.max_depth = max_depth

    def expectiminimax(self, env, depth, player, parent, alpha, beta):
        if env.check_win() or depth == 0:
            return env.evaluate(), None

        if player == env.agent_player:  # Maximizing player
            best_val = -float('inf')
            best_move = None
            legal_moves = env.get_legal_moves(player)
            #print('agent player')
            #print(f'dice: {env.dice_roll}, legal_moves: {legal_moves}')
            curr_dice_roll = env.dice_roll
            for move in legal_moves:
                env.set_dice_roll(curr_dice_roll)
                #print(f'agent curr move: {move}, dice: {env.dice_roll}')
                #print('board before agent making move')
                #print(env.board)
                env.make_simulated_move(player, move)
                #print('board after agent making move')
                #print(env.board)
                val, _ = self.expectiminimax(env, depth - 1, 'chance', player, alpha, beta)
                env.undo_simulated_move()
                if val > best_val:
                    best_val = val
                    best_move = move
                alpha = max(alpha, best_val)
                if beta <= alpha:
                    break
            return best_val, best_move

        elif player == env.get_opponent(env.agent_player):  # Minimizing player
            worst_val = float('inf')
            worst_move = None
            legal_moves = env.get_legal_moves(player)
            #print('opponent player')
            #print(f'dice: {env.dice_roll}, legal_moves: {legal_moves}')
            curr_dice_roll = env.dice_roll
            for move in legal_moves:
                env.set_dice_roll(curr_dice_roll)
                #print(f'opponent curr move: {move}, dice: {env.dice_roll}')
                #print('board before opponent making move')
                #print(env.board)
                env.make_simulated_move(player, move)
                #print('board after opponent making move')
                #print(env.board)
                val, _ = self.expectiminimax(env, depth - 1, 'chance', player, alpha, beta)
                env.undo_simulated_move()
                if val < worst_val:
                    worst_val = val
                    worst_move = move
                beta = min(beta, worst_val)
                if beta <= alpha:
                    break
            return worst_val, worst_move

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

    def choose_move(self, env):
        _, chosen_move = self.expectiminimax(env, self.max_depth, env.agent_player, None, -float('inf'), float('inf'))
        return chosen_move


if __name__ == "__main__":
    
    num_simulations = 1000
    win_count = 0

    for seed in tqdm(range(num_simulations)):
        #print(f'===============seed: {seed}=============')
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
            #print(f'======step_count: {step_count}=======')
            #print(f'dice: {env.dice_roll}')
            #print('board')
            #print(env.board)
            #action = env.action_space.sample()
            action = agent.choose_move(env)
            #env.dice_roll = env.original_dice_roll
            env.set_dice_roll(obs['dice_roll'])
            obs, reward, done, trunc, info = env.step(action)
            if done:
                #print(info)
                if info['message'] == 'You won!':
                    win_count += 1
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
