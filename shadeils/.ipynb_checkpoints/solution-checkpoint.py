import numpy as np
import copy as cp
import torch
import torch.nn as nn
from math import sqrt
from collections import OrderedDict
from .pytorch_utils import pytorch_model_set_weights, pytorch_model_set_weights_from

class Solution:
    
    def create(self):
        pass
    
    def fitness(self, initial_sol):
        pass
    
    def fitness_all(self, population):
        pass
    
    def get_bounds(self):
        pass
    
    def get_partial_bounds(self):
        pass
    
    def clip(self):
        raise NotImplemented("Method 'clip' not implemented.")
        
    def get_model(self):
        raise NotImplemented("Method 'get_model' not implemented.")

class SphereSolution(Solution):
    def __init__(self, dim, lower, upper):
        self.dim = dim
        self.lower = lower
        self.upper = upper
    
    def create(self):
        return np.ones(self.dim)*((self.lower+self.upper)/2.0)
    
    def fitness(self, x):
        return sqrt((x*x).sum())
    
    def fitness_all(self, population):
        return [self.fitness(ind) for ind in population]
    
    def get_bounds(self):
        return list(zip(np.ones(self.dim)*self.lower, np.ones(self.dim)*self.upper))
    
    def get_partial_bounds(self):
        return list(zip(np.ones(self.dim)*self.lower, np.ones(self.dim)*self.upper))

class FullNeuralNetSolution(Solution):
    def __init__(self, model, cross_entropy, inputs, outputs):
        self.model = model
        self.cross_entropy = cross_entropy
        self.inputs = inputs
        self.labels = outputs
    
    def create(self):
        model_dict = self.model.state_dict()
        tensor_dict = OrderedDict()

        for layer_name in model_dict:
            if 'weight' in layer_name:
                size = model_dict[layer_name].size()
                w = torch.empty(size)
                nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
                tensor_dict[layer_name] = w
            if 'bias' in layer_name:
                size = model_dict[layer_name].size()
                b = torch.empty(size)
                torch.nn.init.zeros_(b)
                tensor_dict[layer_name] = w
        
        return tensor_dict
    
    def create_v2(self):
        model_dict = self.model.state_dict()
        tensor_dict = OrderedDict()

        for layer_name in model_dict:
            if 'weight' in layer_name:
                size = model_dict[layer_name].size()
                w = torch.empty(size)
                nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
                tensor_dict[layer_name] = w
            if 'bias' in layer_name:
                size = model_dict[layer_name].size()
                b = torch.empty(size)
                torch.nn.init.zeros_(b)
                tensor_dict[layer_name] = w
        
        return tensor_dict
    
    def fitness(self, x):
        assert isinstance(x, OrderedDict)
        
        clone = cp.deepcopy(self.model)
        pytorch_model_set_weights(clone, x)
        
        outputs = clone(self.inputs)
        loss = self.cross_entropy(outputs, self.labels)
        return loss.item()
    
    def fitness_all(self, population):
        return [self.fitness(ind) for ind in population]
    
    def get_bounds(self):
        dim = len(self.model.state_dict())
        return list(zip(np.ones(dim)*-1, np.ones(dim)*1))
    
    def get_partial_bounds(self):
        dim = len(self.model.state_dict())
        return list(zip(np.ones(dim)*-1, np.ones(dim)*1))
    
    def clip(self, dictionary):
        for name in dictionary:
            dictionary[name].data = tensor.clip(dictionary[name].data, -1, 1)
        
        return dictionary
    
class DownLayerSolution(Solution):
    def __init__(self, model, cross_entropy, inputs, outputs):
        self.model = model
        self.cross_entropy = cross_entropy
        self.inputs = inputs
        self.labels = outputs
        self.target = None
    
    def create(self):
        assert self.target is not None
        model_dict = self.model.state_dict()
        
        for layer_name in model_dict:
            if "weight" in self.target and layer_name == self.target:
                size = model_dict[layer_name].size()
                w = torch.empty(size)
                nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
                return w
            if "bias" in self.target and layer_name == self.target:
                size = model_dict[layer_name].size()
                b = torch.empty(size)
                torch.nn.init.zeros_(b)
                return b
        
        raise Error(f"Target not found {self.target}")
    
    def fitness(self, x):
        assert isinstance(x, torch.Tensor)
        
        clone = cp.deepcopy(self.model)
        pytorch_model_set_weights_from(clone, self.target, x)
        
        outputs = clone(self.inputs)
        loss = self.cross_entropy(outputs, self.labels)
        return loss.item()
    
    def fitness_all(self, population):
        return [self.fitness(ind) for ind in population]
    
    def get_bounds(self):
        dim = len(self.model.state_dict())
        return list(zip(np.ones(dim)*-1, np.ones(dim)*1))
    
    def get_partial_bounds(self):
        dim = len(self.model.state_dict())
        return list(zip(np.ones(dim)*-1, np.ones(dim)*1))
    
    def clip(self, tensor):
        return torch.clamp(tensor, min=-0.99, max=0.99)
    
    def set_target(self, name):
        self.target = name
    
    def get_model(self):
        return self.model
    