import random
import math



class MCTSNode:
    def __init__(self, state, player, parent=None, move=None):
        assert state is not None, "State should not be None"
        self.state = state
        self.player = player
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0

    def add_child(self, child_state, move):
        next_player = self.state.get_opponent(self.player)
        child_node = MCTSNode(child_state, next_player, parent=self, move=move)
        self.children.append(child_node)

    def update(self, result):
        self.visits += 1
        self.wins += result

    def is_fully_expanded(self):
        if self.state is None or self.state.check_win():
            return True
        return len(self.children) == len(self.state.get_legal_actions(self.player))

    def best_child(self, exploration_constant=1.41):
        best_score = float('-inf')
        best_children = []
        for child in self.children:
            if child.visits == 0:
                score = float('inf')  
            else:
                exploit = child.wins / child.visits
                explore = exploration_constant * math.sqrt(math.log(self.visits) / child.visits)
                score = exploit + explore

            if score == best_score:
                best_children.append(child)
            elif score > best_score:
                best_children = [child]
                best_score = score

        if not best_children:
            return random.choice(self.children) if self.children else None
        return random.choice(best_children)


def ucth_search(initial_state, initial_player, iterations, exploration_constant=1.41):
    root = MCTSNode(initial_state, initial_player)

    for _ in range(iterations):
        node = root

        while node is not None and not node.is_fully_expanded():
            if node.state.check_win():
                break
            possible_moves = node.state.get_legal_actions(node.player)
            for move in possible_moves:
                node_state_copy=copy.deepcopy(node.state)
                new_state = node_state_copy.step(move)
                node.add_child(new_state, move)
            bestnode = node.best_child()



        result = simulate_game(bestnode.state)


        while node is not None:
            node.update(result)
            node = node.parent


    best_child = max(root.children, key=lambda c: c.wins / c.visits if c.visits > 0 else 0, default=None)
    return best_child.move 



# def simulate_game(state):
#     current_state = state.copy()
#     while not current_state.check_win():
#         possible_moves = current_state.get_legal_actions(current_state.current_player)
#         move = random.choice(possible_moves)
#         current_state = current_state.make_simulated_action(current_state.current_player, move)
#     return current_state.evaluate()