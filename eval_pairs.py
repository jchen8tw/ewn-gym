import subprocess

policies = ['random', 'minimax', 'mcts']
eval_num = 1024

max_depth = 5 # use in minimax




if __name__ == '__main__':

    for agent in policies:
        for opponent in policies:
            print(f'===== {agent} vs. {opponent} =====')
            command = (
                    f'python3 eval_{agent}.py --num {eval_num} --opponent_policy {opponent} '
                    f'--max_depth {max_depth} --cube_layer 3 --board_size 5 --significance_level 0.05 '
                    )
            command_list = command.split()
            subprocess.run(command_list)
            print()
