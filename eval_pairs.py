import subprocess

policies = [
        'random', 
        'minimax', 
        'mcts', 
        #'alpha_zero',
        ]

eval_num = 1024
board_size = 9
cube_layer = 3
significance_level = 0.05

max_depth = 5 # for minimax
num_env_copies = 5
num_simulations_per_env = 10
model_name = 'checkpoint_242.pth.tar' # for alpha zero



if __name__ == '__main__':
    
    for agent in policies:
        for opponent in policies:
            
            if not(agent == 'mcts' or opponent == 'mcts'):
                continue

            print(f'===== {agent} vs. {opponent} =====')
            command = (
                    f'python3 eval_{agent}.py --num {eval_num} --opponent_policy {opponent} '
                    f'--cube_layer {cube_layer} --board_size {board_size} --significance_level {significance_level} '
                    f'--max_depth {max_depth} --num_env_copies {num_env_copies} --num_simulations_per_env {num_simulations_per_env} '
                    f'--model_folder alpha_zero_models --model_name {model_name}'
                    )
            command_list = command.split()
            subprocess.run(command_list)
            print()
    
