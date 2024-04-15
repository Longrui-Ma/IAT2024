from gridworld import *


class DpAgent:
    def __init__(self, mdp_input, epsilon=0.01):
        self.mdp = mdp_input
        self.epsilon = epsilon
        self.values = {s: 0.0 for s in self.mdp.get_states()}  # value function
        self.values_bis = self.values.copy()
        self.state_proba = {s: [0.0] for s in self.mdp.get_states()}  # for convergence plot
        self.iter = 0  # iteration count for convergence

    def get_value(self, s):
        """ Return the value of a specific state s according to value function v """
        if s == (1, 1):
            return
        value = self.values[s]
        return value

    def get_width(self, v, v_bis):
        """ Return the absolute norm between two value functions """
        max_diff = max(abs(v[s] - v_bis[s]) for s in v)
        return max_diff

    def solve(self):
        """ main solving loop """
        for s in self.mdp.get_states():
            self.update(s)
        self.iter = 1
        while self.get_width(self.values, self.values_bis) > self.epsilon:
            self.values_bis = self.values.copy()
            for s in self.mdp.get_states():
                if not self.mdp.is_terminal(s):
                    self.update(s)
            self.iter += 1
        return self.values

    def update(self, s):
        """ Updates the value of a specific state s"""
        best_value = float('-inf')
        for action in self.mdp.get_actions(s):
            total = 0.0
            for next_state, prob in self.mdp.get_transitions(s, action):
                reward = self.mdp.get_reward(s, action, next_state)
                total += prob * (reward + self.mdp.get_discount_factor() * self.values[next_state])
            best_value = max(best_value, total)
        self.values[s] = best_value
        self.state_proba[s].append(best_value)  # for convergence plot


def policy(s, vf):
    max_value = -float('inf')
    max_action = -1
    print(vf)
    for a in mdp.get_actions(s):
        r = 0
        add_val = 0
        s_proba_list = mdp.get_transitions(s, a)
        for s_proba in s_proba_list:
            r += s_proba[1] * mdp.get_reward(s, a, s_proba[0])
            # if s_proba[0] != ('terminal', 'terminal'):
            add_val += s_proba[1] * vf[s_proba[0]]
        if (r + 0.8 * add_val) > max_value:
            max_value = r + 0.8 * add_val
            max_action = a
    print(max_action)
    return max_action


if __name__ == '__main__':
    mdp = GridWorld(width=4, height=4, )
    agent = DpAgent(mdp, 0.01)
    value_func = agent.solve()
    print("Discount factor: ", agent.mdp.get_discount_factor())
    print("---Value function: ")
    # pretty(value_func)
    # mdp.visualise_value_function(agent)

    print("---Convergence: ")
    print("iterations: ", agent.iter)
    x = [i for i in range(len(agent.state_proba[(0, 0)]))]
    # plt_show_list = [(0, 0), (0, 1), (1, 2), (3, 2), (3, 0)]
    # plt_show_list = [(498, 498)]
    plt_show_list = mdp.get_states()
    try:
        plt_show_list.remove(('terminal', 'terminal'))
    except ValueError:
        print("('terminal', 'terminal') not found.")
    # print(plt_show_list)
    for s in plt_show_list:
        plt.plot(x, agent.state_proba[s], label=s)
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

    print("---Policy: ")
    state = mdp.get_initial_state()
    print("current position", state)
    new_state, _ = mdp.execute(state, policy(state, value_func))
    mdp.visualise()
    mdp.initial_state = new_state
    while not mdp.is_terminal(new_state):
        state = mdp.get_initial_state()
        print("current position", state)
        new_state, _ = mdp.execute(state, policy(state, value_func))
        mdp.visualise()
        mdp.initial_state = new_state
