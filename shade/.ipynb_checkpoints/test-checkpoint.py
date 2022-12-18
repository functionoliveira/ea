import sys
import argparse
import pandas as pd
import numpy as np
from collections import OrderedDict
from pytorch_utils import weight_reset
from torch.nn import MSELoss
from model import DummyNeuralNet, data_x, data_y
from torchsummary import summary
from shade import Shade, DownShade, UpShade
from solution import NeuralNetSolution, SingleLayerSolution

fn_loss = MSELoss()

def test(model, optim, input, output, turns, name=""):
    results = OrderedDict()
    predicted = model(input)
    initial_loss = fn_loss(predicted, output)

    for i in range(1, turns+1):
        print(f"Turn {i}")
        model.apply(weight_reset)
        optim.evolve()
        results[f"turn_{i}"] = optim.best_fitness
    
    np_result = np.array(list(results.values()))
    data = {'initial_loss': [initial_loss.item()], 'mean': [np.mean(np_result)], 'best': [np.min(np_result)], 'worst': [np.max(np_result)] }
    benchmark_df = pd.DataFrame(data=data)
    benchmark_df.to_csv(f"./{name}_benchmark.csv")
    turns_df = pd.DataFrame(data={k: [v] for k, v in results.items()})
    turns_df.to_csv(f"./{name}_turns.csv")
    print("The test has been run successfully, the result is available in the file 'banchmark.csv'.")
    
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

    #seeds
    seeds = [23, 45689, 97232447, 96793335, 12345679]
    args = parser.parse_args(args)
    options = { 1: 'SHADE', 2: 'Down-SHADE', 3: 'Up-SHADE' }

    print(f"Test Type: {args.test_type}")
    print(f"Optim: {options[args.optim]}")
    print(f"Turns: {args.turns}")

    model = None
    optim = None
    input = None
    output = None
    turns = args.turns
    popsize = args.popsize
    gen = args.gens
    name = args.identity

    if args.test_type == 1:
        model = DummyNeuralNet()
        summary(model, (10, 10), device="cpu")
        input = data_x
        output = data_y

    if not args.optim in [1, 2, 3]:
        raise ValueError("Parameter 'optim' must be 1, 2 or 3.")

    if args.optim == 1:
        optim = Shade(NeuralNetSolution(model, MSELoss(), data_x, data_y), threshold=1.0e-6, popsize=popsize, generations=gen)

    if args.optim == 2:
        optim = DownShade(SingleLayerSolution(model, MSELoss(), data_x, data_y), threshold=1.0e-6, popsize=popsize, generations=gen)

    if args.optim == 3:
        optim = UpShade(SingleLayerSolution(model, MSELoss(), data_x, data_y), threshold=1.0e-6, popsize=popsize, generations=gen)

    test(model, optim, input, output, turns, name=name)
    

if __name__ == '__main__':
    main(sys.argv[1:])