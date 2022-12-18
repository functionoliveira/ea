from torch.optim import SGD, Adam
from .evolutionary.leea import Leea

def SgdImp(params, lr=0.001, momentum=0, dampening=0, weight_decay=0, nesterov=False):
    return SGD(params, lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)

def AdamImp(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False):
    return Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)

def LeeaImp(model, data=None, device=None, generations=50, pop_size=100):
    return Leea(model, data=data, device=device, generations=generations, pop_size=pop_size)