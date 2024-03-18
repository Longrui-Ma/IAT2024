from q_function import *
import random 
from matplotlib import pyplot as plt

class q_agent:

    mdp=None
    alpha=None
    episodes=1000
    qFunction={}
    epsilon=0.1

    def __init__(self, mdp, episodes, epsilon, alpha=0.1):
        self.mdp = mdp
        self.alpha = alpha
        self.epsilon=epsilon
        self.qFunction=q_function()

    def greedy(self,state):
        r = random.random()
        if (r>self.epsilon):
            return self.qFunction.get_max_q(state,self.mdp.get_actions(state))[0] 
        else:
            return random.choice(self.mdp.get_actions(state)) 

    def solve(self):
        to_plot=[]
        for i in range(self.episodes):
            state = self.mdp.get_initial_state()
            actions = self.mdp.get_actions(state)
            print("episode:",i," value at initial state: ", self.state_value(self.mdp.get_initial_state()),flush=True)
            to_plot.append(self.state_value(self.mdp.get_initial_state()))
            action=self.greedy(state)
            k=0
            while not self.mdp.is_terminal(state):
                (next_state, reward) = self.mdp.execute(state, action)
                actions = self.mdp.get_actions(next_state)
                next_action = self.greedy(next_state)
                q_value = self.qFunction.get_QValue((state, action))
                delta = self.get_delta(reward, q_value, state, next_state)
                self.qFunction.update((state, action), delta)
                state = next_state
                action = next_action
                k+=1
        plt.plot(to_plot)
        #plt.show()
    """ Calculate the delta for the update """

    def get_delta(self, reward, q_value, state, next_state):
        next_state_value = self.state_value(next_state)
        delta = reward + self.mdp.discount_factor * next_state_value - q_value
        return self.alpha * delta

    """ Get the value of a state """

    def state_value(self, state):
        (_, max_q_value) = self.qFunction.get_max_q(state, self.mdp.get_actions(state))
        return max_q_value
