from mdp import *


class dp_agent():
    mdp=None
    ''' add attributes here! '''
    
    def __init__(self,mdp): #and here...
        self.mdp=mdp

    def get_value(self,s,v):
        #return the value of a specific state s according to value function v
        return 0.0
        
    def get_width(self,v,v_bis):
        #return the absolute norm between two value functions

    def solve(self):
        #main solving loop

    def update(self,s):
        #updates the value of a specific state s
