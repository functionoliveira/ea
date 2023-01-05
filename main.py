import os
import sys
import torch
import argparse
import pandas as pd
import numpy as np
from collections import OrderedDict
from utils.pytorch import weight_reset
from torch.nn import MSELoss
from torch.nn.functional import cross_entropy
from pytorch.models import DummyNeuralNet, CNN
from pytorch.dataset import dummy_data_x, dummy_data_y, getTrainLoader
from torchsummary import summary
from neuroevo.shade import Shade
from neuroevo.shadeils import ShadeILS
from neuroevo.nn_shadeils import DownShadeILS, UpShadeILS
from neuroevo.solution import NeuralNetSolution, SingleLayerSolution
from utils.timer import Chronometer
from utils.log import Log, File
from neuroevo.settings import Settings, mnist_digit_recognition_settings

def test(model, optim, input, output, turns, fn_loss=MSELoss(), name=""):
    chron = Chronometer(True)
    
    print('dataset:', len(input))
    
    if not os.path.isdir('./report'):
        os.makedirs('./report')
    
    stats = f'./report/{name}_stats.csv'
    path_2 = f'./report/{name}_turns.csv'
    
    results = OrderedDict()
    predicted = model(input)
    initial_loss = fn_loss(predicted, output)
    try:
        for i in range(1, turns+1):
            chron.set_message(f'Turn {i}')
            chron.start()
            model.apply(weight_reset)
            optim.evolve()
            results[f"turn_{i}"] = optim.best_fitness
            chron.stop()
            chron.reset()
            print()
        
        np_result = np.array(list(results.values()))
        data = {'initial_loss': [initial_loss.item()], 'mean': [np.mean(np_result)], 'best': [np.min(np_result)], 'worst': [np.max(np_result)] }
        benchmark_df = pd.DataFrame(data=data)
        benchmark_df.to_csv(stats)
        turns_df = pd.DataFrame(data={k: [v] for k, v in results.items()})
        turns_df.to_csv(path_2)
    except Exception as e:
        if len(results.values()) > 0:
            np_result = np.array(list(results.values()))
            data = {'initial_loss': [initial_loss.item()], 'mean': [np.mean(np_result)], 'best': [np.min(np_result)], 'worst': [np.max(np_result)] }
            benchmark_df = pd.DataFrame(data=data)
            benchmark_df.to_csv(stats)
            turns_df = pd.DataFrame(data={k: [v] for k, v in results.items()})
            turns_df.to_csv(path_2)
        raise e
    print()
    print("The test has been run successfully, the result is available in the folder 'report'.")
    
def main(args):
    """
    Main program. It uses
    Run DE for experiments. F, CR must be float, or 'n' as a normal
    """
    description = __file__
    parser = argparse.ArgumentParser(description)    
    parser.add_argument("-i", default="", type=str, dest="identity", help='Identity: identifier of test.')
    parser.add_argument("-o", default=1, type=int, dest="optim", help='Algorithm: (1 - Shade, 2 - ShadeILS, 3 - Down ShadeILS, 4 Up ShadeILS)')
    parser.add_argument("-t", default=1, type=int, dest="test_type", help='Test Type: (1 - Dummy, 2 - MNIST digit recognition)')
    parser.add_argument("-n", default=10, type=int, dest="turns", help='Turns: number of time the test will run')
    parser.add_argument("-g", default=100, type=int, dest="gens", help='Generation: number of time the algorithm will run')
    parser.add_argument("-p", default=100, type=int, dest="popsize", help='Popsize: size of population')
    parser.add_argument("-tr", default=0.00001, type=float, dest="threshold", help='Threshold: limit to stop optimization.')
    parser.add_argument("-me", default=20000, type=int, dest="maxevals", help='Max Evals: number max of evalutations (SHADE-ILS).')
    parser.add_argument("-e", default=20, type=int, dest="epochs", help='Epochs: number of times that algorithm DOWN or UP run.')
    parser.add_argument("-d", default=0, type=int, dest="debug", help='Debug: it shows all individual by generation. (0 - deactivated, 1 - activated).')
    parser.add_argument("-l", default=0, type=int, dest="log", help='Log: type of log (0 - no log, 1 - text file).')
    parser.add_argument("-s", default=0, type=int, dest="settings", help='Default settings')

    #seeds
    seeds = [23, 45689, 97232447, 96793335, 12345679]
    args = parser.parse_args(args)
    test_type_labels = {1: 'Dummy', 2: 'MNIST Handwritten digit recognition'}
    options = { 1: 'SHADE', 2: 'SHADE-ILS', 3: 'Down-SHADE-ILS', 4: 'Up-SHADE-ILS' }

    device = None
    model = None
    fn_loss = None
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    if args.settings == 1:
        settings = mnist_digit_recognition_settings
        settings.identity = args.identity
        test_type = 2
        optim = 4
    else:
        settings = Settings()
        settings.popsize = args.popsize
        settings.generation = args.gens
        settings.identity = args.identity
        settings.threshold = args.threshold
        settings.epochs = args.epochs
        settings.maxevals = args.maxevals
        settings.log = Log if args.log == 0 else File
        settings.debug = bool(args.debug)
        test_type = args.test_type
        optim = args.optim
        
    turns = args.turns
        
    print(f"Name: {settings.identity}")
    print(f"Test Type: {test_type_labels[test_type]}")
    print(f"Optimizer: {options[optim]}")
    print(f"Turns: {args.turns}")
    print(f"Pop Size: {settings.popsize}")
    print(f"Max Evals: {settings.maxevals}")
    print(f"Epochs: {settings.epochs}")
    print(f"Debug: {settings.debug}")
    print(f"Log: {settings.log}")
    print(f"Device: {device}")
    
    if test_type == 1:
        model = DummyNeuralNet().to(device)
        summary(model, (10, 10))
        input = dummy_data_x.to(device)
        output = dummy_data_y.to(device)
        fn_loss = MSELoss()
        
    if test_type == 2:
        model = CNN().to(device)
        summary(model, (1, 28, 28))
        fn_loss = cross_entropy
        settings.dataset = next(iter(getTrainLoader(device, 10000)))
        input, output = settings.dataset
        settings.dataloader = getTrainLoader(device, 512)

    if not optim in [1, 2, 3, 4]:
        raise ValueError("Parameter 'optim' must be 1, 2, 3, or 4.")

    if optim == 1:
        optim = Shade(NeuralNetSolution(model, fn_loss, input, output, device=device), threshold=settings.threshold, popsize=settings.popsize, generations=settings.generation, log=settings.log, debug=settings.debug)
        
    if optim == 2:
        optim = ShadeILS(NeuralNetSolution(model, fn_loss, input, output, device=device), maxevals=settings.maxevals, threshold=settings.threshold, popsize=settings.popsize, generations=settings.generation, debug=settings.debug, log=settings.log)

    if optim == 3:
        settings.solution = SingleLayerSolution(model, fn_loss, None, None, device=device)
        optim = DownShadeILS(settings)

    if optim == 4:
        settings.solution = SingleLayerSolution(model, fn_loss, None, None, device=device)
        optim = UpShadeILS(settings)

    test(model, optim, input, output, turns, fn_loss=fn_loss, name=settings.identity)

if __name__ == '__main__':
    main(sys.argv[1:])
