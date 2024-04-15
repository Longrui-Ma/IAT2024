import random
from matplotlib import pyplot as plt
from gridworld import *
from value_function import *


class QAgent:
    def __init__(self, mdp_input, alpha=0.1, epsilon=0.1):
        self.mdp = mdp_input
        self.alpha = alpha  # Learning rate
        self.gamma = self.mdp.get_discount_factor()  # Discount factor
        self.epsilon = epsilon  # Exploration probability
        self.q_values = {}  # Q table
        for state in self.mdp.get_states():
            self.q_values[state] = {}
            for action in self.mdp.get_actions(state):
                self.q_values[state][action] = 0.0
        self.total_rewards = []  # Rewards for each episode

    def greedy(self, s, greedy=True):
        """ greedy functions returning best action to pick in a state s (based on ε-greedy strategy)"""
        # valid_actions = self.mdp.get_actions(s)
        # if not valid_actions:
        #     raise ValueError("No valid actions available for state {}".format(s))
        if not self.q_values[s]:
            return None  # or raise an error if no actions possible
        max_val = max(self.q_values[s].values())
        best_actions = [a for a, q in self.q_values[s].items() if q == max_val]
        best_action = max(self.q_values[s], key=self.q_values[s].get)
        if greedy:  # Exploration
            if random.random() <= self.epsilon:
                # return random.choice([key for key in self.q_values.keys()])
                # return random.choice(best_actions)
                valid_actions = self.mdp.get_actions(s)
                return random.choice(valid_actions) if valid_actions else None
            else:
                return best_action  # Exploitation

    def state_value(self, state):
        """ Get the value of a state """
        action = self.greedy(state)
        return self.q_values[state][action] if not self.mdp.is_terminal(state) else 0.0

    def get_delta(self, reward, q_value, next_state):
        """ Compute delta for the update """
        return reward + self.gamma * self.state_value(next_state) - q_value

    def update_q_value(self, state, action, delta):
        """ Update Q value """
        self.q_values[state][action] += self.alpha * delta

    def get_q_value(self, state, action):
        return self.q_values[state][action]

    def solve(self):
        """ main solving loop """
        state = self.mdp.get_initial_state()
        # total_reward = 0
        while not self.mdp.is_terminal(state):
            action = self.greedy(state)
            next_state, reward = self.mdp.execute(state, action)
            delta = self.get_delta(reward, self.q_values[state][action], next_state)
            self.update_q_value(state, action, delta)
            state = next_state
        #     total_reward += reward
        # self.total_rewards.append(total_reward)
        return self.q_values

    def get_policy(self):
        """ Get policy """
        policy = {state: self.greedy(state) for state in self.mdp.get_states()}
        return policy


if __name__ == '__main__':
    mdp = GridWorld(width=400, height=400, )
    agent = QAgent(mdp, alpha=0.1, epsilon=0.1)
    value_func = agent.solve()
    print("Discount factor: ", agent.mdp.get_discount_factor())
    alpha_init = 0.1
    alpha = alpha_init
    episodes = 10000
    position = (2, 3)
    # position = (498, 498)

    # Containers for plotting
    y_up, y_down, y_left, y_right = [], [], [], []
    x = []

    # Iterating over the number of iterations
    for i in range(episodes):
        # print_frequency = 1000 if episodes > 1000 else 10
        qf = agent.solve()  # Solve for one episode at a time
        agent.alpha = max(alpha - alpha_init / episodes, 0.01)  # Decrease alpha but do not let it go below 0.01
        q_values_at_position = agent.q_values[position]
        # if (episodes + 1) % print_frequency == 0:  # print only total 100 times
        #     print(f"Episode: {episodes + 1}: Total reward: {total_reward}")

        if i % 200 == 0:  # Collect data every 200 episodes
            x.append(i)
            y_up.append(q_values_at_position.get('▲', 0))
            y_down.append(q_values_at_position.get('▼', 0))
            y_left.append(q_values_at_position.get('◄', 0))
            y_right.append(q_values_at_position.get('►', 0))

    # Plotting results
    plt.title(f'Q-values at position: {position}')
    plt.plot(x, y_up, label='▲')
    plt.plot(x, y_down, label='▼')
    plt.plot(x, y_left, label='◄')
    plt.plot(x, y_right, label='►')
    plt.legend()
    plt.show()
    pretty(qf)
    vf = value_function(qf)
    # Visualize the learned Q-function
    mdp.visualise_q_function(agent)
