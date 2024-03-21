import random 
from matplotlib import pyplot as plt

class q_agent:

    mdp=None
    '''
    add attributes here!
    '''

    def __init__(self, mdp):# and here...
        self.mdp = mdp

    def greedy(self,s):
        #greedy functions returning best action to pick in a state s
        return -1

    def solve(self):
        #main solving loop
        return

    def get_delta(self, reward, q_value, state, next_state):
        #Calculate the delta for the update
        return 0.0


    def state_value(self, state):
        #Get the value of a state
        return 0.0
