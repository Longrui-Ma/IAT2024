class policy():
    
    vf={}
    mdp=None

    def __init__(self,v,mdp):
        self.vf=v.copy()
        self.mdp=mdp

    def select_action(self,s):
        best_a=self.mdp.UP
        val_max=-float('inf')
        for a in self.mdp.get_actions():
            val_a=0
            for next_state,proba in self.mdp.get_transitions(s,a):
                val_a+= proba*(self.mdp.get_reward(s,a,next_state)+self.mdp.get_discount_factor()*self.vf[next_state])
            if (val_a>val_max):
                val_max=val_a
                best_a=a
        return best_a 

