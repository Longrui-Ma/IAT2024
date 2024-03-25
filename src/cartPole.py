import gym
import numpy as np
import random


env = gym.make('CartPole-v1',render_mode="human")

print("action space:", env.action_space) #Discrete(n) : l'ensemble des actions est fini, et il y a n actions : 0,...,n-1

def custom_policy(state):
    tab=[0,1]
    return random.choice(tab)

initial_state=env.reset()
done=False
new_state=initial_state
while not done: 
    new_state,reward,done,_,_ = env.step(custom_policy(new_state))
    print("new_state,reward,done",new_state,reward,done)
    
