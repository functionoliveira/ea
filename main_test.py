import os
import sys
import torch
import argparse
import pandas as pd
import numpy as np
from collections import OrderedDict
from shade.pytorch_utils import weight_reset
from torch.nn import MSELoss
from torch.nn.functional import cross_entropy
from pytorch.models import DummyNeuralNet, CNN
from pytorch.dataset import dummy_data_x, dummy_data_y, getTrainLoader
from torchsummary import summary
from shade.shade import ShadeILS, Shade, DownShade, UpShade, DownShadeILS, UpShadeILS
from shade.solution import NeuralNetSolution, SingleLayerSolution
from timer import Chronometer
from shade.log import Log, File

def test(model, optim, input, output, turns, fn_loss=MSELoss(), name=""):
    chron = Chronometer(True)
    
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
    parser.add_argument("-o", default=1, type=int, dest="optim", help='Algorithm: (1 - Shade, 2- DownShade, 3 - UpShade)')
    parser.add_argument("-t", default=1, type=int, dest="test_type", help='Test Type')
    parser.add_argument("-n", default=10, type=int, dest="turns", help='Turns: number of time the test will run')
    parser.add_argument("-g", default=100, type=int, dest="gens", help='Generation: number of time the algorithm will run')
    parser.add_argument("-p", default=100, type=int, dest="popsize", help='Popsize: size of population')
    parser.add_argument("-tr", default=0.00001, type=float, dest="threshold", help='Threshold: limit to stop optimization.')
    parser.add_argument("-me", default=20000, type=int, dest="maxevals", help='Max Evals: number max of evalutations (SHADE-ILS).')
    parser.add_argument("-e", default=20, type=int, dest="epochs", help='Epochs: number of times that algorithm DOWN or UP run.')
    parser.add_argument("-d", default=0, type=int, dest="debug", help='Debug: it shows all individual by generation. (0 - deactivated, 1 - activated).')
    parser.add_argument("-l", default=0, type=int, dest="log", help='Log: type of log (0 - no log, 1 - text file).')

    #seeds
    seeds = [23, 45689, 97232447, 96793335, 12345679]
    args = parser.parse_args(args)
    options = { 1: 'SHADE', 2: 'Down-SHADE', 3: 'Up-SHADE', 4: 'SHADE-ILS', 5: 'Down-SHADE-ILS', 6: 'Up-SHADE-ILS' }

    print(f"Test Type: {args.test_type}")
    print(f"Optim: {options[args.optim]}")
    print(f"Turns: {args.turns}")

    device = None
    
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")
          
    model = None
    optim = None
    input = None
    output = None
    turns = args.turns
    popsize = args.popsize
    gen = args.gens
    name = args.identity
    fn_loss = None
    threshold=args.threshold
    maxevals=args.maxevals
    epochs=args.epochs
    log_class = None
    debug=bool(args.debug)

    if args.test_type == 1:
        model = DummyNeuralNet().to(device)
        summary(model, (10, 10))
        input = dummy_data_x.to(device)
        output = dummy_data_y.to(device)
        fn_loss = MSELoss()
        
    if args.test_type == 2:
        model = CNN().to(device)
        summary(model, (1, 28, 28))
        input, output = next(iter(getTrainLoader(device)))
        fn_loss = cross_entropy

    if not args.optim in [1, 2, 3, 4, 5, 6]:
        raise ValueError("Parameter 'optim' must be 1, 2, 3, 4, 5 or 6.")

    if args.log == 0:
        log_class = Log
    else:
        log_class = File

    if args.optim == 1:
        optim = Shade(NeuralNetSolution(model, fn_loss, input, output, device=device), threshold=threshold, popsize=popsize, generations=gen, log=log_class, debug=debug)

    if args.optim == 2:
        optim = DownShade(SingleLayerSolution(model, fn_loss, input, output, device=device), threshold=threshold, popsize=popsize, generations=gen)

    if args.optim == 3:
        optim = UpShade(SingleLayerSolution(model, fn_loss, input, output, device=device), threshold=threshold, popsize=popsize, generations=gen)

    if args.optim == 4:
        optim = ShadeILS(NeuralNetSolution(model, fn_loss, input, output, device=device), maxevals=maxevals, threshold=threshold, popsize=popsize, generations=gen, debug=debug, log=log_class)

    if args.optim == 5:
        optim = DownShadeILS(SingleLayerSolution(model, fn_loss, input, output, device=device), maxevals=maxevals, threshold=threshold, popsize=popsize, generations=gen, debug=debug, log=log_class)

    if args.optim == 6:
        optim = UpShadeILS(SingleLayerSolution(model, fn_loss, input, output, device=device), epochs=epochs, maxevals=maxevals, threshold=threshold, popsize=popsize, generations=gen, debug=debug, log=log_class)

    test(model, optim, input, output, turns, fn_loss=fn_loss, name=name)

if __name__ == '__main__':
    main(sys.argv[1:])
