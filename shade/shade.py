import numpy as np
import shade.messages as messages
from shade.evolutionary import EA, CurrentToPBestOneBin
from shade.solution import Solution, SingleLayerSolution
from shade.utils import raise_if, remove, random_indexes

class Shade(EA):
    """Success-History Based Parameter Adaptation for Differential Evolution

    Args:
        EA (Object): Classe base para algoritmos de evolução
    """
    
    def __init__(self, solution, threshold=0.01, generations=10, popsize=100, H=100):
        """Constructor, verify if types of parameters are correct

        Args:
            solution (Solution): Domain of problem solution
            popsize (int): Population size
            H (int): History size
        """
        raise_if(not isinstance(solution, Solution), messages.SOLUTION_TYPE_ERROR, TypeError)
        raise_if(popsize < 10, messages.SOLUTION_TYPE_ERROR, ValueError)

        self.solution = solution
        self.popsize = popsize
        self.H = H
        self.G = range(1, generations)
        self.threshold = threshold
        self.mutation_method = CurrentToPBestOneBin()

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

        return Fnew, CRnew

    def evolve(self):
        # G = 0 - Starting
        population = self.solution.initialize_population(self.popsize)
        population_fitness = self.solution.fitness_all(population)
        MemF = np.ones(self.H)*0.5
        MemCr = np.ones(self.H)*0.5
        # Optional external archive, it used to maintain diversity. SHADE strategy inherits from JADE.
        A = []
        # Index counter
        k = 1
        # 
        pmin = 2 / self.popsize
        # G = Generation array, g = current generation
        for g in self.G:
            # Array of parameters control mutation and crossover operators
            Scr, Sf, weights, offspring = [], [], [], []
            # population = current population, i = individual from current population
            for id, ind in enumerate(population):
                # random selection
                r = np.random.randint(1, self.H)
                # random normal distribution with mean = MemCr and variance = 0.1
                Cr = np.random.randn() * 0.1 + MemCr[r]
                # random cauchy distribution with mean = MemFr and variance = 0.1
                F = np.random.standard_cauchy() * 0.1 + MemF[r]
                # pi = rand[pmin, 0.2] 
                p = np.random.rand() * (0.2-pmin) + pmin
                # Get one random value from population
                r1 = random_indexes(1, self.popsize, ignore=[id])
                xr1 = population[r1]
                # Get one random value from archive
                xr2 = []
                if A:
                    r2 = random_indexes(1, len(A), ignore=[])
                    xr2 = A[r2]
                # Get one random value from p best values
                maxbest = int(p*self.popsize)
                bests = np.argsort(population_fitness)[:maxbest]
                pbest = np.random.choice(bests)
                pbest = population[pbest]
                # Generating mutant
                trial = self.mutation_method(ind, pbest, xr1, xr2, F)
                
                # Calculating fitness
                fitness_t = self.solution.fitness(trial)
                fitness_i = population_fitness[id]
                
                # Creating offspring based on fitness function
                if fitness_t <= fitness_i:
                    offspring.append(trial)
                else:
                    offspring.append(ind)
                    
                # Archiving individuals from old generation
                if fitness_t < fitness_i:
                    A.append(ind)
                    Scr.append(Cr)
                    Sf.append(F)
                    weights.append(fitness_i - fitness_t)
            
            population = offspring
            population_fitness = self.solution.fitness_all(population)
            self.best = population[np.argmin(population_fitness)]
            self.best_fitness = np.min(population_fitness)
            # Remove random items if archive size is bigger than population size
            qtd = len(A) - self.popsize
            if(qtd > 0):
                indexes = np.random.randint(0, len(A), size=qtd)
                remove(A, indexes)
            
            if len(Scr) > 0 and len(Sf) > 0:
                new_F, new_Cr = self.updateMemory(Sf, Scr, weights)
                MemCr[k] = new_Cr
                MemF[k] = new_F
                k = 1 if k >= self.H else k + 1
            
            if self.best_fitness <= self.threshold:
                break

class DownShade(Shade):
    """Success-History Based Parameter Adaptation for Differential Evolution

    Args:
        EA (Object): Classe base para algoritmos de evolução
    """
    
    def __init__(self, solution, epochs=20, threshold=0.01, generations=10, popsize=100, H=100):
        """Constructor, verify if types of parameters are correct

        Args:
            solution (Solution): Domain of problem solution
            popsize (int): Population size
            H (int): History size
        """
        raise_if(not isinstance(solution, SingleLayerSolution), messages.SOLUTION_VALUE_ERROR, ValueError)
        super().__init__(solution, threshold, generations, popsize, H)
        self.epochs = epochs

    def get_layers(self):
        return [idx for idx, _ in enumerate(self.solution.model.parameters())]

    def evolve(self):
        EPOCHS = [e for e in range(self.epochs)]
        layers = self.get_layers()

        for _ in EPOCHS:
            for l in layers:
                self.solution.set_target(l)
                super().evolve()    
                if (self.solution.current_best is None or self.best_fitness < self.solution.current_best_fitness):
                    self.solution.set_current_best(self.best)
                    self.solution.set_current_best_fitness(self.best_fitness)

class UpShade(DownShade):
    def get_layers(self):
        layers = [idx for idx, _ in enumerate(self.solution.model.parameters())]
        layers.reverse()
        return layers
