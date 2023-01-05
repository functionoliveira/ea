import os
import json
import numpy as np
import neuroevo.messages as messages
from neuroevo.ea import EA
from neuroevo.mutation import CurrentToPBestOneBin
from neuroevo.solution import Solution
from utils.general import raise_if, random_indexes, pad_left
from utils.log import Log

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
                ind['layers'] = [{'shape': shape, 'values': tensor.numpy().tolist()} for shape, tensor in i.layers.values()]
                
                if not os.path.isdir(path):
                    os.makedirs(path)
                if best_id == id:
                    file = open(f'{path}/best_{id}_topology.json', 'w')
                else:
                    file = open(f'{path}/ind_{id}_topology.json', 'w')
                
                file.write(json.dumps(ind))
                file.close()

    def evolve(self):
        padding = len(list(str(self.G[-1])))
        self.log.info(f"Starting algorithm={self.__class__.__name__} popsize={self.popsize} generations={len(self.G)} threshold={self.threshold} mutation={self.mutation_method.__class__.__name__}")
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
                xr2 = A[r2].value if A else 0
                # Get one random value from p best values
                maxbest = int(p*self.popsize)
                bests = np.argsort(self.population_fitness)[:maxbest]
                pbest = np.random.choice(bests)
                pbest = self.population[pbest]
                # Generating mutant
                trial = self.mutation_method(ind.value, pbest.value, xr1.value, xr2, F)
                # Calculating fitness
                fitness_t = self.solution.fitness(ind, trial)
                fitness_i = self.population_fitness[id]
                
                self.log.info(f"algorithm={self.__class__.__name__} generation={g} Cr={Cr} F={F} pi={p} r1={r1} r2={r2} maxbest={maxbest} fitness_trial={fitness_t} fitness_ind={fitness_i}")
                
                # Archiving individuals from old generation
                if fitness_t < fitness_i:
                    offspring.append(self.solution.update_chromosome(ind, trial))
                    A.append(ind)
                    Scr.append(Cr)
                    Sf.append(F)
                    weights.append(fitness_i - fitness_t)
                else:
                    offspring.append(ind)
            
            self.population = offspring
            self.population_fitness = self.solution.fitness_all(self.population)
            self.best_id = np.argmin(self.population_fitness)
            self.best = self.population[self.best_id]
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
            self.compound_folder.append(f'gen_{pad_left(g, padding)}') 
            self.save_generation()
            self.compound_folder.pop()
            
            if self.best_fitness <= self.threshold:
                break
