from gridworld import *

mdp = GridWorld()

print("states:",mdp.get_states())
print("terminal states:",mdp.get_goal_states())
print("actions:",mdp.get_actions())

def policy_custom(state):
    return mdp.UP

while(1):
    state=mdp.get_initial_state()
    new_state,_ = mdp.execute(state,mdp.UP)
    mdp.initial_state=new_state
    #print(mdp.get_transitions(mdp.initial_state,mdp.UP))
    mdp.visualise()
