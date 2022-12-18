import torch
import torch.nn as nn
import torch.nn.functional as F

class DummyNeuralNet(nn.Module):
    def __init__(self):
        super(DummyNeuralNet, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(10, 15),
            nn.ReLU(),
            nn.Linear(15, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.linear(x)

class WeightClipper(object):
    def __init__(self, frequency=1):
        self.frequency = frequency

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(-1,1)
        if hasattr(module, 'bias'):
            w = module.bias.data
            w = w.clamp(-1,1)

class CNN(nn.Module):
    def __init__(self): 
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 28, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(28, 14, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(14, 7, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(28, 128),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(128, 80),
            nn.ReLU()
        )


        self.fc3 = nn.Sequential(
            nn.Linear(80, 10),
            nn.Softmax()
        )

    def remove_softmax(self):
        self.fc3 =  nn.Sequential(*list(self.fc3.children())[:-1])

    def reset_weights(self):
        self.apply(weights_init)

    def clip(self):
        clipper = WeightClipper()
        self.apply(clipper)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

class LeNet5(nn.Module):
    def __init__(self, n_classes):
        super(LeNet5, self).__init__()
        
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=n_classes),
        )

    def clip(self):
        clipper = WeightClipper()
        self.apply(clipper)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return logits
