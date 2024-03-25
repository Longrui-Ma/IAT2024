class value_function():
    vf={}

    def __init__(self,v):
        self.vf=v.copy()

    def get_value(self,t):
        if t in self.vf:
            return self.vf[t]
        else:
            return -float("inf")

