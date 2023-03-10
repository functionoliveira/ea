import torch
import numpy as np
import copy as cp
import torch.nn as nn
from collections import OrderedDict
from shade.pytorch_utils import pytorch_model_set_weights, pytorch_model_set_weights_from

class Solution:
    def __init__(self):
        raise RuntimeError("Abstract class cannot be instanciated.")
    
    def create(self):
        pass
    
    def initialize_population(self):
        raise NotImplementedError("Method 'initialize_population' is not implemented.")
    
    def fitness(self, initial_sol):
        raise NotImplementedError("Method 'fitness' is not implemented.")
    
    def fitness_all(self, population):
        raise NotImplementedError("Method 'fitness_all' is not implemented.")
    
    def get_bounds(self):
        pass
    
    def get_partial_bounds(self):
        pass
    
    def clip(self):
        raise NotImplemented("Method 'clip' not implemented.")
        
    def get_model(self):
        raise NotImplemented("Method 'get_model' not implemented.")
    
# Unimodal function
class SphereSolution(Solution):
    def __init__(self, dim, lower, upper):
        self.dim = dim
        self.lower = lower
        self.upper = upper
    
    def initialize_population(self, popsize):
        return [np.random.random_integers(self.lower, self.upper, self.dim) * np.random.rand(self.dim) for _ in range(popsize)]
    
    def fitness(self, x):
        return (x*x).sum()
    
    def fitness_all(self, population):
        return [self.fitness(ind) for ind in population]
    
class RosenbrockSolution(Solution):
    def __init__(self, dim, lower, upper):
        self.dim = dim
        self.lower = lower
        self.upper = upper

    def initialize_population(self, popsize):
        return [np.random.random_integers(self.lower, self.upper, self.dim) * np.random.rand(self.dim) for _ in range(popsize)]
    
    def fitness(self, x):
        stop = len(x) / 2
        sum = 0 
        for i in range(1, len(x)):
            sum += 100 * (((x[2*i - 2] ** 2) - x[2*i - 1]) ** 2) + (x[2 * i - 2] - 1) ** 2
            
            if i >= stop:
                break
        
        return sum
    
    def fitness_all(self, population):
        return [self.fitness(ind) for ind in population]
    
class RosenbrockTensorSolution(Solution):
    def __init__(self, dim, lower, upper):
        self.dim = dim
        self.lower = lower
        self.upper = upper

    def initialize_population(self, popsize):
        return [nn.Conv2d(1, 28, kernel_size=self.dim).weight.detach().numpy() for _ in range(popsize)]
    
    def fitness(self, x):
        x = x.flatten()
        stop = len(x) / 2
        sum = 0 
        for i in range(1, len(x)):
            sum += 100 * (((x[2*i - 2] ** 2) - x[2*i - 1]) ** 2) + (x[2 * i - 2] - 1) ** 2
            
            if i >= stop:
                break

        return sum
    
    def fitness_all(self, population):
        return [self.fitness(ind) for ind in population]
    
class NeuralNetSolution(Solution):
    def __init__(self, model, fn_loss, input, output, device=None):
        self.model = model
        self.fn_loss = fn_loss
        self.input = input
        self.output = output
        self.device = device

    def initialize_population(self, popsize):
        population = []
        
        for _ in range(popsize):
            ind = OrderedDict()
            for idx, data in enumerate(self.model.parameters()):
                if idx % 2 == 0:
                    size = data.size()
                    w = torch.empty(size)
                    nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
                    ind[idx] = w.numpy()
                if idx % 2 == 1:
                    size = data.size()
                    b = torch.empty(size)
                    torch.nn.init.zeros_(b)
                    ind[idx] = b.numpy()
            population.append(ind)
        
        return population
    
    def fitness(self, x):
        assert isinstance(x, OrderedDict)
        
        clone = cp.deepcopy(self.model)
        pytorch_model_set_weights(clone, x)
        clone.to(self.device)
        predicted = clone(self.input)
        loss = self.fn_loss(predicted, self.output)
        return loss.item()
    
    def fitness_all(self, population):
        return [self.fitness(ind) for ind in population]

class SingleLayerSolution(Solution):
    def __init__(self, model, fn_loss, input, output, device=None, target=None):
        self.model = model
        self.fn_loss = fn_loss
        self.input = input
        self.output = output
        self.target = target
        self.current_best = None
        self.current_best_fitness = None
        self.device = device

    def set_target(self, value):
        self.target = value
        
    def set_current_best(self, value):
        self.current_best = value
        
    def set_current_best_fitness(self, value):
        self.current_best_fitness = value

    def initialize_population(self, popsize):
        if self.target is None:
            raise ValueError("Attribute 'target' cannot be None.")
            
        population = []
        
        for _ in range(popsize):
            ind = OrderedDict()
            for idx, data in enumerate(self.model.parameters()):
                if idx != self.target:
                    ind[idx] = data.cpu().detach().numpy()
                if idx % 2 == 0 and idx == self.target:
                    size = data.size()
                    w = torch.empty(size)
                    nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
                    ind[idx] = w.numpy()
                if idx % 2 == 1 and idx == self.target:
                    size = data.size()
                    b = torch.empty(size)
                    torch.nn.init.zeros_(b)
                    ind[idx] = b.numpy()
            population.append(ind)
            
        if (self.current_best is not None):
            idx = np.random.randint(0, popsize)
            population[idx] = self.current_best
            
        return population
    
    def fitness(self, x):
        if self.target is None:
            raise ValueError("Attribute 'target' cannot be None.")
        assert isinstance(x, OrderedDict)

        pytorch_model_set_weights(self.model, x)
        self.model.to(self.device)
        predicted = self.model(self.input)
        loss = self.fn_loss(predicted, self.output)
        return loss.item()
    
    def fitness_all(self, population):
        return [self.fitness(ind) for ind in population]
