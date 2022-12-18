import numpy as np
from collections import OrderedDict

class Mutation:
    def __call__(self):
        raise NotImplementedError("Method 'generate' is not implemented.")

class Crossover:
    pass

class Selection:
    pass
    
class CurrentToPBestOneBin(Mutation):
    
    def __call__(self, xi, xbest, xr1, xr2, F):
        if isinstance(xi, OrderedDict):
            result = OrderedDict()
            
            for name in xi:
                if len(xr2) > 0:
                    result[name] =  xi[name] + F * (xbest[name] - xi[name]) + F * (xr1[name]-xr2[name])
                else:
                    result[name] =  xi[name] + F * (xbest[name] - xi[name])
                    
            return result
        
        if len(xr2) > 0:
            return xi + F * (xbest - xi) + F * (xr1-xr2)
        else:
            return xi + F * (xbest - xi)

class EA:
    """
    Classe abstrata para construção de algoritmos neuro evolutivos.
    
    """
    
    def __init__(self):
        pass
    
    def evolve(self):
        raise NotImplementedError("Method 'evolve' is not implemented.")
