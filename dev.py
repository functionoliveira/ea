from shade.solution import SingleLayerSolution
from torch.nn.functional import cross_entropy
from datasets.mnist_handwritter_digit_recognition import getTrainLoader
from sklearn.model_selection import train_test_split
from models import CNN
import numpy as np

model = CNN()
input, output = next(iter(getTrainLoader(None, 10000)))
s = SingleLayerSolution(model, cross_entropy, input, output, device=None)

print(len(input))

pop = s.initialize_population(10)
s.set_target(0)
flatten = s.to_1d_array(pop[0]) * 0.1
print(flatten)
dictionary = s.to_solution(flatten) 
print(dictionary)