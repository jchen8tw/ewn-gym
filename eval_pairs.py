import subprocess

policies = ['random', 'minimax', 'mcts']
eval_num = 100





if __name__ == '__main__':

    for agent in policies:
        for opponent in policies:
            print(f'====={agent} vs. {opponent}=====')
            command = f'python3 eval_{agent}.py --num {eval_num} --opponent_policy {opponent}'
            command_list = command.split()
            subprocess.run(command_list)
