import torch
import numpy as np
import copy as cp
import torch.nn as nn
from collections import OrderedDict, namedtuple
from shade.pytorch_utils import pytorch_model_set_weights
from pytorch.models import CNN

class Solution:
    def __init__(self):
        raise RuntimeError("Abstract class cannot be instanciated.")
    
    def initialize_population(self):
        raise NotImplementedError("Method 'initialize_population' is not implemented.")
    
    def fitness(self, initial_sol):
        raise NotImplementedError("Method 'fitness' is not implemented.")
    
    def fitness_all(self, population):
        raise NotImplementedError("Method 'fitness_all' is not implemented.")
    
    def set_data(self, input, output):
        raise NotImplementedError("Method 'set_data' is not implemented.")
    
    def set_target(self, value):
        raise NotImplementedError("Method 'set_target' is not implemented.")
    
    def set_current_id(self):
        raise NotImplemented("Method 'set_current_id' not implemented.")
    
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
                    ind[idx] = w.numpy().astype(dtype=np.double)
                if idx % 2 == 1:
                    size = data.size()
                    b = torch.empty(size)
                    torch.nn.init.zeros_(b)
                    ind[idx] = b.numpy().astype(dtype=np.double)
            population.append(ind)
        
        return population
    
    def to_1d_array(self, x):
        arr = []
        
        for i in x.values():
            for j in i.flatten():
                arr.append(j)
    
        return np.array(arr)
    
    def to_solution(self, oned_array):
        layers = OrderedDict()
        i = 0
        size = 0
        for idx, data in enumerate(self.model.parameters()):
            shape = list(data.size())
            size += np.prod(shape)
            arr = []
            while i < size:
                arr.append(oned_array[i])
                i += 1
            layers[idx] = np.reshape(np.array(arr, dtype=np.double), shape)
    
        return layers
    
    def fitness(self, x):
        assert isinstance(x, OrderedDict)
        
        clone = cp.deepcopy(self.model)
        pytorch_model_set_weights(clone, x)
        clone.to(self.device)
        predicted = clone(self.input.double())
        loss = self.fn_loss(predicted, self.output)
        return loss.item()
    
    def fitness_all(self, population):
        return [self.fitness(ind) for ind in population]
    
    def ls_fitness(self, x):
        assert isinstance(x, np.ndarray)
        
        clone = cp.deepcopy(self.model)
        pytorch_model_set_weights(clone, self.to_solution(x))
        clone.to(self.device)
        predicted = clone(self.input.double())
        loss = self.fn_loss(predicted, self.output)
        return loss.item()

Layer = namedtuple("Layer", "shape tensor")
Chromosome = namedtuple("Chromosome", "value layers")

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

    def set_data(self, input, output):
        self.input = input
        self.output = output

    def set_target(self, value):
        self.target = value
    
    def create_chromosome(self):
        layers = OrderedDict()
        for id, data in enumerate(self.model.parameters()):
            size = data.size()
            tensor_empty = torch.empty(size)
            if id % 2 == 0:
                nn.init.xavier_uniform_(tensor_empty, gain=nn.init.calculate_gain('relu'))
            if id % 2 == 1:
                torch.nn.init.trunc_normal_(tensor_empty, mean=-0.5, std=0.5, a=-1, b=0)
            layers[id] = Layer(size, tensor_empty)
        value = layers[self.target].tensor.numpy().flatten()
        return Chromosome(value, layers)
    
    def update_chromosome(self, chromosome, val):
        layers = OrderedDict()
        
        for id, layer in enumerate(chromosome.layers.values()):
            if id == self.target:
                layers[id] = Layer(layer.shape, torch.from_numpy(np.reshape(np.array(val, dtype=np.double), layer.shape)))
            else:
                layers[id] = layer
        
        return Chromosome(val, layers)
    
    def reload_chromosome(self, layer, population):
        new_pop = []
        self.target = layer
        self.l_shape = list(self.model.parameters())[layer].size()
        
        for c in population:
            new_c = Chromosome(c.layers[layer].tensor.numpy().flatten(), c.layers)
            new_pop.append(new_c)

        return new_pop
    
    def initialize_population(self, popsize):
        return [self.create_chromosome() for _ in range(popsize)]

    def get_shape(self):
        return list(self.l_shape)

    def fitness(self, chromosome, val=None):
        assert isinstance(chromosome, Chromosome)

        for id, param in enumerate(self.model.parameters()):
            if not val is None and id == self.target:
                param.data = torch.from_numpy(np.reshape(np.array(val, dtype=np.double), self.l_shape))
            else:
                param.data = chromosome.layers[id].tensor.double()

        self.model.to(self.device)
        predicted = self.model(self.input.double())
        loss = self.fn_loss(predicted, self.output)
        return loss.item()
    
    def fitness_all(self, population):
        return [self.fitness(c) for c in population]
