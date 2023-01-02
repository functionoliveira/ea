import os
import json
import numpy as np
import shade.messages as messages
from shade.evolutionary import EA, CurrentToPBestOneBin
from shade.solution import Solution, SingleLayerSolution
from shade.utils import raise_if, remove, random_indexes
from shade.mechanism.crossover import SADECrossover
from shade.mechanism.pool import PoolLast
from enum import Enum
from scipy.optimize import fmin_l_bfgs_b
from shade.mts import mtsls
from shade.log import Log

class Shade(EA):
    """Success-History Based Parameter Adaptation for Differential Evolution

    Args:
        EA (Object): Classe base para algoritmos de evolução
    """
    
    def __init__(self, solution, threshold=0.01, generations=10, popsize=100, debug=True, log=Log, population=list(), population_fitness=list()):
        """Constructor, verify if types of parameters are correct

        Args:
            solution (Solution): Domain of problem solution
            popsize (int): Population size
            H (int): History size
        """
        raise_if(not isinstance(solution, Solution), messages.SOLUTION_TYPE_ERROR, TypeError)
        raise_if(popsize < 10, messages.SOLUTION_TYPE_ERROR, ValueError)

        self.population = population
        self.population_fitness = population_fitness
        self.solution = solution
        self.popsize = popsize
        self.H = popsize
        self.G = range(1, generations+1)
        self.threshold = threshold
        self.mutation_method = CurrentToPBestOneBin()
        self.crossover = SADECrossover(2)
        self.debug = debug
        self.root_folder = 'shade_output'
        self.compound_folder = []
        self.log = log('./shade_output')

    def updateMemory(self, Sf, SCr, improvements):
        """
        Update the new F and CR using the new Fs and CRs, and its improvements
        """
        total = np.sum(improvements)
        assert total > 0
        weights = improvements/total

        Fnew = np.sum(weights*Sf*Sf)/np.sum(weights*Sf)
        Fnew = np.clip(Fnew, 0, 1)
        CRnew = np.sum(weights*SCr)
        CRnew = np.clip(CRnew, 0, 1)

        self.log.info(f"algorithm={self.__class__.__name__} F_new={Fnew} Cr_new={CRnew}")

        return Fnew, CRnew

    def save_generation(self):
        if len(self.compound_folder) > 0:
            path = f'./{self.root_folder}/{"/".join(self.compound_folder)}'
        else:
            path = f'./{self.root_folder}'
        
        best_id = np.argmin(self.population_fitness)
        if self.debug == True:
            for id, i in enumerate(self.population):
                ind = {}                
                ind['id'] = id
                ind['fitness'] = self.population_fitness[id]
                ind['layers'] = [{'shape': j.shape, 'values': j.tolist()} for j in i.values()]
                
                if not os.path.isdir(path):
                    os.makedirs(path)
                if best_id == id:
                    file = open(f'{path}/best_{id}_topology.json', 'w')
                else:
                    file = open(f'{path}/ind_{id}_topology.json', 'w')
                
                file.write(json.dumps(ind))
                file.close()

    def evolve(self):
        self.log.info(f"Starting algorithm={self.__class__.__name__} popsize={self.popsize} generations={len(self.G)} threshold={self.threshold} mutation={self.mutation_method.__class__.__name__} crossover={self.crossover.__class__.__name__}")
        # G = 0 - Starting
        if len(self.population) == 0:
            self.population = self.solution.initialize_population(self.popsize)
        if len(self.population_fitness) == 0:
            self.population_fitness = self.solution.fitness_all(self.population)
        
        if not hasattr(self, 'MemF'):
            self.MemF = np.ones(self.H)*0.5
        if not hasattr(self, 'MemCr'):
            self.MemCr = np.ones(self.H)*0.5
        # Optional external archive, it used to maintain diversity. SHADE strategy inherits from JADE.
        A = []
        # Index counter
        k = 1
        # 
        pmin = 2 / self.popsize
        self.log.info(f"algorithm={self.__class__.__name__} pmin={pmin} best_fitness={np.min(self.population_fitness)}")
        # G = Generation array, g = current generation
        for g in self.G:
            # Array of parameters control mutation and crossover operators
            Scr, Sf, weights, offspring = [], [], [], []
            # population = current population, i = individual from current population
            for id, ind in enumerate(self.population):
                # random selection
                r = np.random.randint(1, self.H)
                # random normal distribution with mean = MemCr and variance = 0.1
                Cr = np.random.randn() * 0.1 + self.MemCr[r]
                # random cauchy distribution with mean = MemFr and variance = 0.1
                F = np.random.standard_cauchy() * 0.1 + self.MemF[r]
                # pi = rand[pmin, 0.2] 
                p = np.random.rand() * (0.2-pmin) + pmin
                # Get one random value from population
                r1 = random_indexes(1, self.popsize, ignore=[id])
                xr1 = self.population[r1]
                # Get one random value from archive
                r2 = random_indexes(1, len(A), ignore=[]) if A else -1
                xr2 = A[r2] if A else []
                # Get one random value from p best values
                maxbest = int(p*self.popsize)
                bests = np.argsort(self.population_fitness)[:maxbest]
                pbest = np.random.choice(bests)
                pbest = self.population[pbest]
                # Generating mutant
                trial = self.mutation_method(ind, pbest, xr1, xr2, F)
                # Calculating fitness
                fitness_t = self.solution.fitness(trial)
                fitness_i = self.population_fitness[id]
                
                self.log.info(f"algorithm={self.__class__.__name__} generation={g} Cr={Cr} F={F} pi={p} r1={r1} r2={r2} maxbest={maxbest} fitness_trial={fitness_t} fitness_ind={fitness_i}")
                
                # Creating offspring based on fitness function
                # options = [ind, trial]
                # fitness = [fitness_i, fitness_t]
                # offspring.append(options[np.argmin(fitness)])
                    
                
                # Archiving individuals from old generation
                if fitness_t < fitness_i:
                    offspring.append(trial)
                    A.append(ind)
                    Scr.append(Cr)
                    Sf.append(F)
                    weights.append(fitness_i - fitness_t)
                else:
                    offspring.append(ind)
            
            self.population = offspring
            self.population_fitness = self.solution.fitness_all(self.population)
            self.best = self.population[np.argmin(self.population_fitness)]
            self.best_fitness = np.min(self.population_fitness)
            # Remove random items if archive size is bigger than population size
            qtd = len(A) - self.popsize
            if(qtd > 0):
                for _ in range(qtd):
                    A.pop(np.random.randint(0, len(A)))
            
            if len(Scr) > 0 and len(Sf) > 0:
                new_F, new_Cr = self.updateMemory(Sf, Scr, weights)
                self.MemCr[k] = new_Cr
                self.MemF[k] = new_F
                k = 1 if k >= self.H - 1 else k + 1
            
            self.log.info(f"algorithm={self.__class__.__name__} generation={g} Archive={len(A)} Scr={len(Scr)} Sf={len(Sf)} avg_weights={np.average(weights)} avg_fitness={np.average(self.population_fitness)} best_fitness={np.min(self.population_fitness)} fitness={self.population_fitness}")
            # Save info about generation in json files
            self.compound_folder.append(f'gen_{g}') 
            self.save_generation()
            self.compound_folder.pop()
            
            if self.best_fitness <= self.threshold:
                break

class DownShade(Shade):
    """Success-History Based Parameter Adaptation for Differential Evolution

    Args:
        EA (Object): Classe base para algoritmos de evolução
    """
    
    def __init__(self, solution, epochs=20, threshold=0.01, generations=10, popsize=100):
        """Constructor, verify if types of parameters are correct

        Args:
            solution (Solution): Domain of problem solution
            popsize (int): Population size
            H (int): History size
        """
        raise_if(not isinstance(solution, SingleLayerSolution), messages.SOLUTION_VALUE_ERROR, ValueError)
        super().__init__(solution, threshold, generations, popsize)
        self.epochs = epochs
        self.root_folder = 'down_shade_output'
        self.compound_folder = ['', '', '']

    def get_layers(self):
        return [idx for idx, _ in enumerate(self.solution.model.parameters())]

    def evolve(self):
        EPOCHS = [e for e in range(self.epochs)]
        layers = self.get_layers()

        for e in EPOCHS:
            self.compound_folder[0] = f'epoch_{e}'
            for l in layers:
                self.compound_folder[1] = f'layer_{l}'
                self.solution.set_target(l)
                super().evolve()    
                if (self.solution.current_best is None or self.best_fitness < self.solution.current_best_fitness):
                    self.solution.set_current_best(self.best)
                    self.solution.set_current_best_fitness(self.best_fitness)

class UpShade(DownShade):
    def __init__(self, solution, epochs=20, threshold=0.01, generations=10, popsize=100):
        raise_if(not isinstance(solution, SingleLayerSolution), messages.SOLUTION_VALUE_ERROR, ValueError)
        super().__init__(solution, epochs, threshold, generations, popsize)
        self.root_folder = 'up_shade_output'
    
    def get_layers(self):
        layers = [idx for idx, _ in enumerate(self.solution.model.parameters())]
        layers.reverse()
        return layers

class LS_METHOD(Enum):
    ALL = 1
    MTS = 2
    GRAD = 3
    
    def equals(self, other):
        return (self.__class__ == other.__class__ and
            self.value == other.value)

    def is_all(self):
        return self == self.ALL
    
    def is_mts(self):
        return self == self.MTS
    
    def is_grad(self):
        return self == self.GRAD

class ShadeILS(Shade):
        
    def __init__(self, solution, maxevals=10000, threshold=0.01, generations=10, popsize=100, debug=True, log=Log):
        super().__init__(solution, threshold, 10, popsize, debug, log)
        
        self.maxevals = maxevals
        self.current_best = None
        self.best_global = None
        self.SR_MTS = None
        self.SR_global_mts = None
        self.ls_method = LS_METHOD.ALL
        self.root_folder = 'shadeils_output'
        self.log = log('./shadeils_output')
        self.num_worse = 0
        self.num_restarts = 0

    def set_current_best(self, solution, fitness):
        if self.current_best is None or self.current_best_fitness > fitness:
            self.current_best = solution
            self.current_best_fitness = fitness
        
    def set_best_global(self, solution, fitness):
        if self.best_global is None or self.best_global_fitness > fitness:
            self.best_global = solution
            self.best_global_fitness = fitness
            
        if not fitness in self.population_fitness:
            worst_id = np.argmax(self.population_fitness)
            self.population[worst_id] = solution
            self.population_fitness[worst_id] = fitness

    def apply_localsearch(self, name, maxevals):
        if self.ls_method.is_grad():
            x0 = self.solution.to_1d_array(self.current_best)
            sol, fit, info = fmin_l_bfgs_b(self.solution.ls_fitness, x0=x0, approx_grad=1, maxfun=maxevals, disp=False)
            sol = self.solution.to_solution(sol)
            funcalls = info['funcalls']
        elif self.ls_method.is_mts():
            if name.lower() == "global":
                SR = self.SR_global_mts
            else:
                SR = self.SR_MTS

            x0 = self.solution.to_1d_array(self.current_best)
            res, self.SR_MTS = mtsls(self.solution.ls_fitness, x0, self.current_best_fitness, 0, 1, maxevals, SR)
            sol = self.solution.to_solution(res.solution)
            fit = res.fitness
            funcalls = self.maxevals
        else:
            raise NotImplementedError(f"Method '{self.ls_method}' is not supported.")

        self.set_current_best(sol, fit)
        self.totalevals += funcalls
            
    def get_ratio_improvement(self, previous_fitness, new_fitness):
        if previous_fitness == 0:
            improvement = 0
        else:
            improvement = (previous_fitness-new_fitness)/previous_fitness

        return improvement

    def set_region_ls(self):
        self.SR_MTS = np.copy(self.SR_global_mts)

    def reset_ls(self):
        self.SR_global_mts = np.ones(self.solution.to_1d_array(self.current_best).shape)*0.5
        self.SR_MTS = self.SR_global_mts

    def evolve(self):
        self.log.info(f"Starting algorithm={self.__class__.__name__} popsize={self.popsize} generations={len(self.G)} max_evals={self.maxevals} threshold={self.threshold} mutation={self.mutation_method.__class__.__name__} crossover={self.crossover.__class__.__name__}")
        self.totalevals = 1
        
        if len(self.population) == 0:
            self.population = self.solution.initialize_population(self.popsize)
        if len(self.population_fitness) == 0:    
            self.population_fitness = self.solution.fitness_all(self.population)
        best_id = np.argmin(self.population_fitness)

        initial_sol = self.population[best_id]
        initial_fitness = self.population_fitness[best_id]

        self.set_current_best(initial_sol, initial_fitness)
        self.set_best_global(initial_sol, initial_fitness)
               
        apply_de, apply_ls = (True, True)
    
        self.reset_ls()
        methods = [LS_METHOD.MTS, LS_METHOD.GRAD]

        pool_global = PoolLast(methods)
        pool = PoolLast(methods)

        evals_gs = 20
        evals_ls = 10
        previous_fitness = 0
        g = 0
        
        self.log.info(f"algorithm={self.__class__.__name__} initial_fitness={initial_fitness} evals_gs={evals_gs} evals_ls={evals_ls}")
        
        while self.totalevals < self.maxevals:
            method = None
            if not pool_global.is_empty():
                previous_fitness = self.current_best_fitness
                self.ls_method = pool_global.get_new()
                self.apply_localsearch("global", evals_gs)
                improvement = self.get_ratio_improvement(previous_fitness, self.current_best_fitness)
                pool_global.improvement(self.ls_method, improvement, 2)
                self.set_best_global(self.current_best, self.current_best_fitness)
                self.log.info(f"algorithm={self.__class__.__name__} phase='Global Search' previous_fitness={previous_fitness} ls_method={self.ls_method} improvement={improvement} total_evals={self.totalevals} current_best={self.current_best_fitness} best_global={self.best_global_fitness}")

            self.set_region_ls()
            method = pool.get_new()

            if apply_de:
                self.compound_folder.append(f'turn_{g}')
                super().evolve()
                improvement = self.current_best_fitness - self.best_fitness
                self.totalevals += len(self.G)
                self.set_current_best(self.best, self.best_fitness)
                self.log.info(f"algorithm={self.__class__.__name__} phase='DE' improvement={improvement} total_evals={self.totalevals} current_best={self.current_best_fitness} best_global={self.best_global_fitness}")

            if apply_ls:
                previous_fitness = self.current_best_fitness
                self.apply_localsearch("local", evals_ls)
                improvement = self.get_ratio_improvement(previous_fitness, self.current_best_fitness)
                pool.improvement(method, improvement, 10, .25)
                self.log.info(f"algorithm={self.__class__.__name__} phase='Local Search' previous_fitness={previous_fitness} ls_method={method} improvement={improvement} total_evals={self.totalevals} current_best={self.current_best_fitness} best_global={self.best_global_fitness}")

            self.set_best_global(self.current_best, self.current_best_fitness)
            
            # Restart if it is not improved
            if (previous_fitness == 0):
                ratio_improvement = 1
            else:
                ratio_improvement = (previous_fitness - self.best_global_fitness) / previous_fitness

            if ratio_improvement >= self.threshold:
                self.num_worse = 0
            else:
                self.num_worse += 1
                # Random the LS
                self.reset_ls()
            
            self.log.info(f"algorithm={self.__class__.__name__} best_global={self.best_global_fitness} total_evals={self.totalevals} ratio_improvement={ratio_improvement} num_worse={self.num_worse}")
            
            # restart criteria
            if self.num_worse >= 5:
                self.num_worse = 0
                # Increase a 1% of values
                posi =  np.random.choice(self.popsize)
                surviver = self.population[posi]

                # Init DE
                self.population = self.solution.initialize_population(self.popsize)
                self.population[posi] = surviver
                self.population_fitness = self.solution.fitness_all(self.population)
                self.totalevals += self.popsize
                best_id = np.argmin(self.population_fitness)

                initial_sol = self.population[best_id]
                initial_fitness = self.population_fitness[best_id]
                
                self.current_best = initial_sol
                self.current_best_fitness = initial_fitness
                
                self.best_global = initial_sol
                self.best_global_fitness = initial_fitness

                # Random the LS
                pool_global.reset()
                pool.reset()
                self.reset_ls()
                self.num_restarts += 1
                self.log.info(f"algorithm={self.__class__.__name__} phase='Restart' restarts={self.num_restarts} best_global={self.best_global_fitness} total_evals={self.totalevals}")
            
            self.save_generation()
            self.compound_folder.pop()
            g += 1
        
class DownShadeILS(ShadeILS):
    """Success-History Based Parameter Adaptation for Differential Evolution

    Args:
        EA (Object): Classe base para algoritmos de evolução
    """
    
    def __init__(self, solution, epochs=20, maxevals=10000, threshold=0.01, generations=10, popsize=100, debug=True, log=Log):
        raise_if(not isinstance(solution, SingleLayerSolution), messages.SOLUTION_VALUE_ERROR, ValueError)
        super().__init__(solution, maxevals, threshold, generations, popsize, debug, log)
        self.epochs = epochs
        self.log = log('./down_shadeils_output')
        self.root_folder = 'down_shadeils_output'
        self.compound_folder = ['', '']

    def get_layers(self):
        return [idx for idx, _ in enumerate(self.solution.model.parameters())]

    def evolve(self):
        self.log.info(f"Starting algorithm={self.__class__.__name__} popsize={self.popsize} generations={len(self.G)} max_evals={self.maxevals} epochs={self.epochs} threshold={self.threshold} mutation={self.mutation_method.__class__.__name__} crossover={self.crossover.__class__.__name__}")
        EPOCHS = [e for e in range(self.epochs)]
        layers = self.get_layers()

        for e in EPOCHS:
            self.compound_folder[0] = f'epoch_{e}'
            for l in layers:
                self.compound_folder[1] = f'layer_{l}'
                self.solution.set_target(l)
                super().evolve()
                # if (self.solution.current_best is None or self.best_global_fitness < self.solution.current_best_fitness):
                #     self.solution.set_current_best(self.best_global)
                #     self.solution.set_current_best_fitness(self.best_global_fitness)
                self.log.info(f"algorithm={self.__class__.__name__} epoch={e} layer={l} best_fitness={self.best_global_fitness}")
            self.log.info(f"algorithm={self.__class__.__name__} epoch={e} best_fitness={self.best_global_fitness}")

class UpShadeILS(DownShadeILS):
    def __init__(self, solution, epochs=20, maxevals=10000, threshold=0.01, generations=10, popsize=100, debug=True, log=Log):
        raise_if(not isinstance(solution, SingleLayerSolution), messages.SOLUTION_VALUE_ERROR, ValueError)
        super().__init__(solution, epochs, maxevals, threshold, generations, popsize, debug, log)
        self.log = log('./up_shadeils_output')
        self.root_folder = 'up_shadeils_output'
    
    def get_layers(self):
        layers = [idx for idx, _ in enumerate(self.solution.model.parameters())]
        layers.reverse()
        return layers

    def evolve(self):
        super().evolve()
