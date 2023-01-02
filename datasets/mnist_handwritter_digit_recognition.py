from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data.dataloader import default_collate

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
transform2 = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
resize = transforms.Compose([transforms.Resize((32, 32)),
                                 transforms.ToTensor()])

def getTrainLoader(device, batch_size=64):
    if device != None:
        return DataLoader(MNIST('./files/', train=True, transform=resize, download=True), batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))
    else:
        return DataLoader(MNIST('./files/', train=True, transform=resize, download=True), batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))
    
def getValidationLoader(device, batch_size):
    if device != None:
        return DataLoader(MNIST('./files/', train=False, transform=resize, download=True), batch_size=batch_size, shuffle=True, collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x)))
    else:
        return DataLoader(MNIST('./files/', train=False, transform=resize, download=True), batch_size=batch_size, shuffle=True)