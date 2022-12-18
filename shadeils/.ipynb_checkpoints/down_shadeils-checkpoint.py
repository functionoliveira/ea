import torch
import collections
import numpy as np
import torch.nn as nn
from shadeils.crossover import Crossover, SADECrossover
from shadeils.shadeils import PoolLast, applySHADE, applySHADE_v2, apply_localsearch, apply_localsearch_v2, get_ratio_improvement, check_evals, reset_de, reset_de_v2, reset_ls, set_region_ls
from shadeils.de_utils import EAresult
from shadeils.function import Evaluator
from .solution import Solution, SphereSolution
from .pytorch_utils import tensor2npArray, orderedDict2npArray, pytorch_model_set_weights_by_name

class ShadeILS:
    def __init__(self, solution, info, evals, sys_out, popsize=100, crossover=SADECrossover(2), threshold=0.1):
        assert isinstance(solution, Solution)
        assert isinstance(crossover, Crossover)
        self.solution = solution
        self.crossover = crossover
        self.info = info
        self.evals = evals
        self.popsize = popsize
        self.sys_out = sys_out
        self.threshold = threshold
    
    def fit(self):
        lower = self.info['lower']
        upper = self.info['upper']
        evals = self.evals[:]

        initial_sol = self.solution.create()
        current_best_fitness = self.solution.fitness(initial_sol)

        maxevals = evals[-1]
        totalevals = 1
        
        bounds = self.solution.get_bounds()
        bounds_partial = self.solution.get_partial_bounds()
        
        popsize = self.popsize
        population = reset_de_v2(self.popsize, self.solution, 100)
        population_fitness = self.solution.fitness_all(population)
        best_id = np.argmin(population_fitness)
        
        initial_fitness = current_best_fitness
        
        if initial_fitness < population_fitness[best_id]:
            #self.sys_out.write("Best initial_sol\n")
            population[best_id] = initial_sol
            population_fitness[best_id] = initial_fitness

        current_best = EAresult(solution=population[best_id], fitness=population_fitness[best_id], evaluations=totalevals)
        
        crossover = self.crossover
        best_global_solution = current_best.solution
        best_global_fitness = current_best.fitness
        current_best_solution = best_global_solution

        apply_ls = False
        apply_de = True
        applyDE = applySHADE
        
        reset_ls(2, lower, upper)
        methods = ['mts', 'grad']

        pool_global = PoolLast(methods)
        pool = PoolLast(methods)

        num_worse = 0

        evals_gs = min(50*2, 25000)
        evals_de = min(50*2, 25000)
        evals_ls = min(10*2, 5000)
        num_restarts = 0

        while totalevals < maxevals:
            method = ""

            if not pool_global.is_empty():
                previous_fitness = current_best.fitness
                method_global = pool_global.get_new()
                
                # Transforming orderedDict in numpy array to work in scipy lib
                #np_current_best_solution = orderedDict2npArray(current_best_solution)
                #current_best = apply_localsearch_v2("Global", method_global, self.solution.fitness, bounds, np_current_best_solution, current_best.fitness, evals_gs, self.sys_out)
                #totalevals += current_best.evaluations
                #improvement = get_ratio_improvement(previous_fitness, current_best.fitness)

                #pool_global.improvement(method_global, improvement, 2)
                #evals = check_evals(totalevals, evals, current_best.fitness, best_global_fitness, self.sys_out)
                #current_best_solution = current_best.solution
                #current_best_fitness = current_best.fitness

                if current_best_fitness < best_global_fitness:
                     best_global_solution = np.copy(current_best_solution)
                     best_global_fitness = self.solution.fitness(best_global_solution)

                for i in range(1):
                    current_best = EAresult(solution=current_best_solution, fitness=current_best_fitness, evaluations=0)
                    set_region_ls()

                    method = pool.get_new()

                    if apply_de:
                        result, bestInd = applySHADE_v2(crossover, self.solution, evals_de, population, population_fitness, best_id, current_best, self.sys_out, len(population))
                        improvement = current_best.fitness - result.fitness
                        totalevals += result.evaluations
                        evals = check_evals(totalevals, evals, result.fitness, best_global_fitness, self.sys_out)
                        current_best = result

                    if apply_ls:
                        result = apply_localsearch("Local", method, self.fn_fitness, bounds_partial, current_best.solution, current_best.fitness, evals_ls, self.sys_out)
                        improvement = get_ratio_improvement(current_best.fitness, result.fitness)
                        totalevals += result.evaluations
                        evals = check_evals(totalevals, evals, result.fitness, best_global_fitness, self.sys_out)
                        current_best = result

                        pool.improvement(method, improvement, 10, .25)

                    current_best_solution = current_best.solution
                    current_best_fitness = current_best.fitness

                    if current_best_fitness < best_global_fitness:
                        best_global_fitness = current_best_fitness
                        best_global_solution = np.copy(current_best_solution)

                    # Restart if it is not improved
                    if (previous_fitness == 0):
                        ratio_improvement = 1
                    else:
                        ratio_improvement = (previous_fitness-result.fitness)/previous_fitness

                    #self.sys_out.write("TotalImprovement[{:d}%] {:.5f} => {:.5f} ({})\tRestart: {}\n".format(
                    #    int(100*ratio_improvement), previous_fitness, result.fitness,
                    #    num_worse, num_restarts)
                    #)

                    if ratio_improvement >= self.threshold:
                        num_worse = 0
                    else:
                        num_worse += 1
                        imp_str = ",".join(["{}:{}".format(m, val) for m, val in pool.improvements.items()])
                        #self.sys_out.write("Pools Improvements: {}".format(imp_str))

                        # Random the LS
                        #reset_ls(self.dimension, lower, upper, method)

                    if num_worse >= 3:
                        num_worse = 0
                        #self.sys_out.write("Restart:{0:.5f} for {1:.2f}: with {2:d} evaluations\n".format(current_best.fitness, ratio_improvement, totalevals))
                        # Increase a 1% of values
                        posi =  np.random.choice(self.popsize)
                        new_solution = np.random.uniform(-0.01, 0.01, self.dimension)*(upper-lower)+population[posi]
                        new_solution = np.clip(new_solution, lower, upper)
                        current_best = EAresult(solution=new_solution, fitness=fitness_fun(new_solution), evaluations=0)
                        current_best_solution = current_best.solution
                        current_best_fitness = current_best.fitness

                        # Init DE
                        population = reset_de(self.popsize, self.dimension, lower, upper, self.info_de)
                        populationFitness = [self.fn_fitness(ind) for ind in population]
                        totalevals += self.popsize

                        totalevals += self.popsize
                        # Random the LS
                        pool_global.reset()
                        pool.reset()
                        reset_ls(self.dimension, lower, upper)
                        num_restarts += 1

                    #self.sys_out.write("{0:.5f}({1:.5f}): with {2:d} evaluations\n".format(current_best_fitness, best_global_fitness, totalevals))
                    #self.sys_out.flush()

                    if totalevals >= maxevals:
                        break

        #self.sys_out.write("%f,%s,%d\n" %(abs(best_global_fitness), ' '.join(map(str, best_global_solution)), totalevals))
        #self.sys_out.flush()
        return result

class BackUpShadeILS:
    def __init__(self, fn_fitness, info, dim, evals, fid, info_de, popsize=100, debug=False, threshold=0.05):
        assert isinstance(fn_fitness, Evaluator)
        self.fn_fitness = fn_fitness
        self.info = info
        self.dimension = dim
        self.evals = evals
        self.sys_out = fid
        self.info_de = info_de
        self.popsize = popsize
        self.debug = debug
        self.threshold = threshold
    
    def fit(self):
        lower = self.info['lower']
        upper = self.info['upper']
        evals = self.evals[:]

        initial_sol = np.ones(self.dimension)*((lower+upper)/2.0)
        current_best_fitness = self.fn_fitness(initial_sol)

        maxevals = evals[-1]
        totalevals = 1

        bounds = list(zip(np.ones(self.dimension)*lower, np.ones(self.dimension)*upper))
        bounds_partial = list(zip(np.ones(self.dimension)*lower, np.ones(self.dimension)*upper))

        popsize = self.popsize
        population = reset_de(self.popsize, self.dimension, lower, upper, self.info_de)
        population_fitness = [self.fn_fitness(ind) for ind in population]
        bestId = np.argmin(population_fitness)

        initial_fitness = current_best_fitness

        if initial_fitness < population_fitness[bestId]:
            self.sys_out.write("Best initial_sol\n")
            population[bestId] = initial_sol
            population_fitness[bestId] = initial_fitness

        current_best = EAresult(solution=population[bestId,:], fitness=population_fitness[bestId], evaluations=totalevals)

        crossover = SADECrossover(2)
        best_global_solution = current_best.solution
        best_global_fitness = current_best.fitness
        current_best_solution = best_global_solution

        apply_de = apply_ls = True
        applyDE = applySHADE
  
        reset_ls(self.dimension, lower, upper)
        methods = ['mts', 'grad']

        pool_global = PoolLast(methods)
        pool = PoolLast(methods)

        num_worse = 0

        evals_gs = min(50*self.dimension, 25000)
        evals_de = min(50*self.dimension, 25000)
        evals_ls = min(10*self.dimension, 5000)
        num_restarts = 0

        while totalevals < maxevals:
            method = ""

            if not pool_global.is_empty():
                previous_fitness = current_best.fitness
                method_global = pool_global.get_new()
                current_best = apply_localsearch("Global", method_global, self.fn_fitness, bounds, current_best_solution, current_best.fitness, evals_gs, self.sys_out)
                totalevals += current_best.evaluations
                improvement = get_ratio_improvement(previous_fitness, current_best.fitness)

                pool_global.improvement(method_global, improvement, 2)
                evals = check_evals(totalevals, evals, current_best.fitness, best_global_fitness, self.sys_out)
                current_best_solution = current_best.solution
                current_best_fitness = current_best.fitness

                if current_best_fitness < best_global_fitness:
                     best_global_solution = np.copy(current_best_solution)
                     best_global_fitness = self.fn_fitness(best_global_solution)

                for i in range(1):
                    current_best = EAresult(solution=current_best_solution, fitness=current_best_fitness, evaluations=0)
                    set_region_ls()

                    method = pool.get_new()

                    if apply_de:
                        result, bestInd = applyDE(crossover, self.fn_fitness, self.info, self.dimension, evals_de, population, population_fitness, bestId, current_best, self.sys_out, self.info_de)
                        improvement = current_best.fitness - result.fitness
                        totalevals += result.evaluations
                        evals = check_evals(totalevals, evals, result.fitness, best_global_fitness, self.sys_out)
                        current_best = result

                    if apply_ls:
                        result = apply_localsearch("Local", method, self.fn_fitness, bounds_partial, current_best.solution, current_best.fitness, evals_ls, self.sys_out)
                        improvement = get_ratio_improvement(current_best.fitness, result.fitness)
                        totalevals += result.evaluations
                        evals = check_evals(totalevals, evals, result.fitness, best_global_fitness, self.sys_out)
                        current_best = result

                        pool.improvement(method, improvement, 10, .25)

                    current_best_solution = current_best.solution
                    current_best_fitness = current_best.fitness

                    if current_best_fitness < best_global_fitness:
                        best_global_fitness = current_best_fitness
                        best_global_solution = np.copy(current_best_solution)

                    # Restart if it is not improved
                    if (previous_fitness == 0):
                        ratio_improvement = 1
                    else:
                        ratio_improvement = (previous_fitness-result.fitness)/previous_fitness

                    self.sys_out.write("TotalImprovement[{:d}%] {:.5f} => {:.5f} ({})\tRestart: {}\n".format(
                        int(100*ratio_improvement), previous_fitness, result.fitness,
                        num_worse, num_restarts)
                    )

                    if ratio_improvement >= self.threshold:
                        num_worse = 0
                    else:
                        num_worse += 1
                        imp_str = ",".join(["{}:{}".format(m, val) for m, val in pool.improvements.items()])
                        self.sys_out.write("Pools Improvements: {}".format(imp_str))

                        # Random the LS
                        reset_ls(self.dimension, lower, upper, method)

                    if num_worse >= 3:
                        num_worse = 0
                        self.sys_out.write("Restart:{0:.5f} for {1:.2f}: with {2:d} evaluations\n".format(current_best.fitness, ratio_improvement, totalevals))
                        # Increase a 1% of values
                        posi =  np.random.choice(self.popsize)
                        new_solution = np.random.uniform(-0.01, 0.01, self.dimension)*(upper-lower)+population[posi]
                        new_solution = np.clip(new_solution, lower, upper)
                        current_best = EAresult(solution=new_solution, fitness=fitness_fun(new_solution), evaluations=0)
                        current_best_solution = current_best.solution
                        current_best_fitness = current_best.fitness

                        # Init DE
                        population = reset_de(self.popsize, self.dimension, lower, upper, self.info_de)
                        populationFitness = [self.fn_fitness(ind) for ind in population]
                        totalevals += self.popsize

                        totalevals += self.popsize
                        # Random the LS
                        pool_global.reset()
                        pool.reset()
                        reset_ls(self.dimension, lower, upper)
                        num_restarts += 1

                    self.sys_out.write("{0:.5f}({1:.5f}): with {2:d} evaluations\n".format(current_best_fitness, best_global_fitness, totalevals))
                    self.sys_out.flush()

                    if totalevals >= maxevals:
                        break

        self.sys_out.write("%f,%s,%d\n" %(abs(best_global_fitness), ' '.join(map(str, best_global_solution)), totalevals))
        self.sys_out.flush()
        return result
        #w = torch.empty(size)
        #nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
        #return w

class UpShadeILS(ShadeILS):
    def __init__(self, model):
        super(ShadeILS, self).__init__()
        self.model = model

    def fit(self):
        model_dict = self.model.state_dict()
        print(model_dict)
        with torch.no_grad():
            for layer_name in model_dict:
                if 'weight' in layer_name:
                    #print("weight:", model_dict[layer_name].data)
                    model_dict[layer_name].data = torch.empty(model_dict[layer_name].size()) #super().fit(model_dict[layer_name].size())
                #if 'bias' in layer_name:
                #    print("bias:", model_dict[layer_name].data)
                #    model_dict[layer_name].data = super().fit(model_dict[layer_name].size())
    
class DownShadeILS(ShadeILS):
    def __init__(self, solution, info, evals, sys_out, popsize=100, crossover=SADECrossover(2)):
        super().__init__(solution, info, evals, sys_out, popsize, crossover)

    def fit(self):
        model_dict = collections.OrderedDict(reversed(list(self.solution.get_model().state_dict().items())))
        best = collections.OrderedDict()
        
        with torch.no_grad():
            for layer_name in model_dict:
                self.solution.set_target(layer_name)
                result = super().fit()
                print(result)
                best[layer_name] = result.solution
                
        print(best)
        self.best = best
        
    def test(self):
        pytorch_model_set_weights_by_name(self.solution.get_model(), self.best)
        model = self.solution.get_model()
        output = model(self.solution.inputs)
        loss = self.solution.cross_entropy(output, self.solution.labels)
        print(loss.item())