import torch
import numpy as np
import torch.nn as nn

def glorot_init(shape):
    w = torch.empty(shape)
    nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
    return w

def pytorch_model_set_weights_from(model, target, tensor):
    model_dict = model.state_dict().items()
    
    with torch.no_grad():
        for layer_name in model_dict:
            if layer_name == target:
                model_dict[layer_name].data = tensor
                
def pytorch_model_set_weights(model, values):
    model_dict = model.state_dict()
    
    with torch.no_grad():
        for v_layer, m_layer in zip(values, model_dict):
            model_dict[m_layer].data = values[v_layer]
            
def pytorch_model_set_weights_by_name(model, dictionary):
    model_dict = model.state_dict()
    
    with torch.no_grad():
        for d_layer in dictionary:
            for m_layer in model_dict:
                if d_layer == m_layer:
                    model_dict[m_layer].data = dictionary[d_layer].data
                    break
            
def tensor2npArray(tensor):
    print(type(tensor.numpy()))
    print(tensor.numpy())
    
def orderedDict2npArray(dictionary):
    items = dictionary.items()
    numpy_arr = []
    
    for name in dictionary:
        numpy_arr.append(dictionary[name].numpy().values)
    
    return np.array(numpy_arr)
