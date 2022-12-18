import numpy as np
import copy as cp
from .mechanism import create_population, evaluate_pop_v2, offspring_v4
from .pytorch_utils import pytorch_model_set_weights_v2

class Leea:
    
    def __init__(self, model, data=None, device=None, generations=2000, pop_size=20):
        self.model = model
        self.data = data
        self.device = device
        self.generations = generations
        self.pop_size = pop_size
        self.population = create_population(pop_size, model, device)
        self.decay = 1
        self.reduce_recall_param = 1 / generations
    
    def reduce_decay(self):
        if self.decay > 0:
            self.decay = self.decay - self.reduce_recall_param
    
    def zero_grad(self):
        pass
    
    def step(self):
        inputs, labels = self.data
        fitness = list(evaluate_pop_v2(self.model, inputs, labels, self.population))
        inheritance_fitness = fitness
    
        for gen in range(self.generations):
            (p, f) = offspring_v4(self.model, self.device, inputs, labels, cp.deepcopy(self.population), fitness)
            self.population = p
            fitness = f
            self.reduce_decay()
            
        best_of_all = self.population[np.argmin(fitness)]
        pytorch_model_set_weights_v2(self.model, best_of_all)
