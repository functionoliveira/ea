import time
import torch
from models import CNN, LeNet5
from torch.nn.functional import cross_entropy
from torch.nn import CrossEntropyLoss
from torchsummary import summary
from optimizers.sgd import SgdImp, AdamImp, LeeaImp
from train import SgdTrain, Train
from datasets.mnist_handwritter_digit_recognition import getTrainLoader, getValidationLoader
from shadeils.down_shadeils import ShadeILS, DownShadeILS

import sys
from shadeils.function import sphere, Sphere, NNClassfier
from shadeils.solution import SphereSolution, FullNeuralNetSolution, DownLayerSolution

print("Starting ring society project...")
print("Use CUDA:", torch.cuda.is_available())

device = None
train_loader = getTrainLoader(device)
validation_loader = getValidationLoader(device)

input, output = next(iter(train_loader))

model = CNN().to(torch.device('cuda:0'))
summary(model, (1, 28, 28))
model_dict = model.state_dict()
    
with torch.no_grad():
    for layer_name in model_dict:
        print("Name:", layer_name, " Tensor:", model_dict[layer_name].data)
#a = DownShadeILS(
#  CNN(), 
#  Sphere(),
#  {'lower': -5, 'upper': 5, 'threshold': 0, 'best': 0},
#  2,
#  [2],
#  sys.stdout,
#  100
##)
#a.fit()

#model = ShadeILS(
#    SphereSolution(2, -10, 10),
#    {'lower': -5, 'upper': 5, 'threshold': 0, 'best': 0},
#    [2],
#    100,
#    sys.stdout
#)
#model.fit()

#opti = DownShadeILS(
#    DownLayerSolution(CNN(), cross_entropy, input, output),
#    {'lower': -5, 'upper': 5, 'threshold': 0, 'best': 0},
#    [50],
#    sys.stdout,
#    popsize=5
#)
#opti.fit()
#opti.test()


#NNSolution = ShadeILS(
#    FullNeuralNetSolution(CNN(), cross_entropy, input, output),
#    {'lower': -5, 'upper': 5, 'threshold': 0, 'best': 0},
#    [2],
#    sys.stdout
#)
#NNSolution.fit()



#if torch.cuda.is_available():
#    device = torch.device('cuda:0')
#    le_net_5 = LeNet5(10).to(device)
#    le_net_5.cuda()
#else:
#    le_net_5 = LeNet5(10)

#summary(le_net_5, (1, 32, 32))

#le_net5_sgd = SgdImp(le_net_5.parameters(), lr=0.01)
#le_net5_adam_sgd = AdamImp(le_net_5.parameters(), lr=0.001)
#le_net5_leea = LeeaImp(le_net_5, device=device, generations=10, pop_size=10)

#handler = Train(le_net_5, le_net5_sgd, epochs=20, device=device, train_loader=train_loader, validation_loader=validation_loader)
#handler.train()

#handler = Train(le_net_5, le_net5_adam_sgd, epochs=20, device=device, train_loader=train_loader, validation_loader=validation_loader)
#handler.train()

#handler = Train(le_net_5, le_net5_sgd, epochs=20, device=device, train_loader=train_loader, validation_loader=validation_loader)
#handler.train()
