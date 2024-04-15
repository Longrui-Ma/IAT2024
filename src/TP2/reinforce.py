import gym
import numpy as np

import torch
import torch.nn
import torch.nn.functional
import torch.optim
from torch.distributions import Categorical

import random

seed=42;
env = gym.make('CartPole-v1',render_mode="human")
#env.seed(args.seed)
env.reset(seed=42)
torch.manual_seed(42)


class Policy(torch.nn.Module):# Policy extends torch.nn (neural network) module, i.e. policy is a neural network
    def __init__(self):
        super(Policy, self).__init__()          
        self.affine1 = torch.nn.Linear(4, 128)            
        #must instanciate neural layers, e.g. self.affine1=torch.nn.Linear(n,p)
        #   where n is the size of the input data and p is the size of the output data
        #   tips: -activation functions -> you can use torch.nn.Dropout, torch.nn.softmax

    def forward(self, x):
        #according to the network layer defined in __init__(), perform a forward pass, i.e. compute the output of the neural network,
        #   given the input x
        
        #For example, y = self.affine1(x) computes the matrix product between the layer self.affine and the vector x.
        return self.affine1(x);

policy = Policy() #initiale policy as a neural network
optimiser = torch.optim.Adam(policy.parameters(), lr=1e-2) #optimizer that will be used to perform backward gradient propagation is classical Adam, with learning rate 1e-2

def select_action(state):
    #given a state as an input of your neural network policy, compute the output and select an action based on the output of the neural network
    tab=[0,1]
    return random.choice(tab)

def finish_episode():
    #an episode just finished being sampled according to the policy; perform weights updates here
    return;

def main(rendering,nb_steps,nb_ep):
    for i_episode in range(nb_ep):
        state = env.reset()
        state=state[0]
        for t in range(1, nb_steps):
            action = select_action(state)
            state, reward, done, truncated,info = env.step(action)
            if rendering:
                env.render()
            if done:
                break

        finish_episode()

if __name__=='__main__':
    rendering=True
    nb_ep=1000
    nb_steps=10000
    main(rendering,nb_steps,nb_ep)
