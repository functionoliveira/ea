import torch

def pytorch_model_set_weights(model, tensor_weights):
    model_dict = model.state_dict()
    with torch.no_grad():
        for layer_name in model_dict:
            if 'weight' in layer_name:
                model_dict[layer_name].data = tensor_weights.pop()

def pytorch_model_set_weights_v2(model, tensor_weights):
    for i, param in enumerate(model.parameters()):
        param.data = tensor_weights[i]

                
def pytorch_model_get_weights(model):
    model_dict = model.state_dict()
    with torch.no_grad():
        for layer_name in model_dict:
            if 'weight' in layer_name:
                print(model_dict[layer_name].data)

def pytorch_model_get_weights_v2(model):
        for param in model.parameters():
            print(param.data)
