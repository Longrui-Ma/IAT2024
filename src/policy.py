class policy():
    
    vf={}
    mdp=None

    def __init__(self,v,mdp):
        self.vf=v.copy()
        self.mdp=mdp

    def select_action(self,s):
        #selectionne la meilleure action à faire pour l'état s, étant donné la fonction de valeur en attribut de la classe
        best_a=self.mdp.UP
        return best_a 

