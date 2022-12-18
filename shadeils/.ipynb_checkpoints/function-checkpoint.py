import torch
import copy as cp
from math import sqrt
from torch.nn.functional import cross_entropy

def sphere(x):
    return sqrt((x*x).sum())

def pytorch_model_set_weights(model, target, tensor):
    model_dict = model.state_dict().items()
    
    with torch.no_grad():
        for layer_name in model_dict:
            if layer_name == target:
                model_dict[layer_name].data = tensor
                
def pytorch_model_set_weights(model, values):
    model_dict = model.state_dict().items()
    
    with torch.no_grad():
        for v_layer, m_layer in zip(values, model_dict):
            model_dict[m_layer].data = values[v_layer]

class Evaluator:
    def set_target(self, target_name):
        pass
        
class Sphere(Evaluator):
    def __call__(self, x):
        return sqrt((x*x).sum())

class NNClassfier(Evaluator):
    def __init__(self, model, inputs, labels):
        self.model = model
        self.inputs = inputs
        self.labels = labels
        self.target = None
        
    def set_target(self, target_name):
        self.target = target_name
        
    def __call__(self, value):
        clone = cp.deepcopy(self.model)
        
        if self.target is not None:
            assert isinstance(value, torch.Tensor)
            pytorch_model_set_weights(clone, self.target, value)
            
        if self.target is None:
            assert isinstance(value, dict)
            pytorch_model_set_weights(clone, value)
        
        outputs = clone(self.inputs)
        loss = cross_entropy(outputs, self.labels)
        return loss.item()
