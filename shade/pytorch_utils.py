import torch
import numpy as np
import torch.nn as nn

def glorot_init(shape):
    w = torch.empty(shape)
    nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
    return w

def pytorch_model_set_weights_from(model, target, value):
    for i, param in enumerate(model.parameters()):
        if i == target:
            param.data = torch.from_numpy(value[i])
    
def pytorch_model_set_weights(model, values):        
    for i, param in enumerate(model.parameters()):
        param.data = torch.from_numpy(values[i])
            
def pytorch_model_set_weights_by_name(model, dictionary):
    model_dict = model.state_dict()
    
    with torch.no_grad():
        for d_layer in dictionary:
            for m_layer in model_dict:
                if d_layer == m_layer:
                    model_dict[m_layer].data = dictionary[d_layer].data
                    break

def weight_reset(m):
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()
