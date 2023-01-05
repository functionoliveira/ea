import datetime as dt
import numpy as np
from enum import Enum
from neuroevo.shade import Shade
from neuroevo.pool import PoolLast
from neuroevo.mts import mtsls
from utils.log import Log
from scipy.optimize import fmin_l_bfgs_b

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
    def __init__(self, solution, maxevals=10000, threshold=0.01, generations=10, popsize=100, debug=True, log=Log, identity=''):
        super().__init__(solution, threshold, generations, popsize, debug, log)
        
        self.maxevals = maxevals
        self.current_best = None
        self.best_global = None
        self.SR_MTS = None
        self.SR_global_mts = None
        self.ls_method = LS_METHOD.ALL
        self.root_folder = f'output/shadeils/{identity}' if identity is not None else f'output/shadeils/{dt.datetime.now().strftime("%Y%m%d")}'
        self.log = log(f'./{self.root_folder}')
        self.num_worse = 0
        self.num_restarts = 0

    def get_current_best(self):
        return self.population[np.argmin(self.population_fitness)]

    def get_current_best_fitness(self):
        return np.min(self.population_fitness)

    def set_current_best(self, id, solution, fitness):
        if self.current_best is None or self.current_best_fitness > fitness:
            self.current_best_id = id
            self.current_best = solution
            self.current_best_fitness = fitness
            self.population[id] = solution
            self.population_fitness[id] = fitness
        
    def set_best_global(self, id, solution, fitness):
        if self.best_global is None or self.best_global_fitness > fitness:
            self.best_global_id = id
            self.best_global = solution
            self.best_global_fitness = fitness
            self.population[id] = solution
            self.population_fitness[id] = fitness

    def apply_localsearch(self, name, maxevals):
        # if self.ls_method.is_grad():
        #     sol, fit, info = fmin_l_bfgs_b(self.solution.fitness, x0=self.current_best, approx_grad=1, maxfun=maxevals, disp=False)
        #     funcalls = info['funcalls']
        # elif self.ls_method.is_mts():
        if name.lower() == "global":
            SR = self.SR_global_mts
        else:
            SR = self.SR_MTS

        res, self.SR_MTS = mtsls(self.solution.fitness, self.get_current_best(), self.get_current_best_fitness(), -500, 500, maxevals, SR)
        sol = self.solution.update_chromosome(self.get_current_best(), res.solution)
        fit = res.fitness
        funcalls = self.maxevals
        # else:
        #     raise NotImplementedError(f"Method '{self.ls_method}' is not supported.")

        self.set_current_best(self.current_best_id, sol, fit)
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
        self.SR_global_mts = np.ones(np.prod(self.solution.get_shape()))*0.5
        self.SR_MTS = self.SR_global_mts

    def evolve(self):
        self.log.info(f"Starting algorithm={self.__class__.__name__} popsize={self.popsize} generations={len(self.G)} max_evals={self.maxevals} threshold={self.threshold} mutation={self.mutation_method.__class__.__name__}")
        self.totalevals = 1
        
        if len(self.population) == 0:
            self.population = self.solution.initialize_population(self.popsize)
        if len(self.population_fitness) == 0:    
            self.population_fitness = self.solution.fitness_all(self.population)
        best_id = np.argmin(self.population_fitness)

        initial_sol = self.population[best_id]
        initial_fitness = self.population_fitness[best_id]

        self.set_current_best(best_id, initial_sol, initial_fitness)
        self.set_best_global(best_id, initial_sol, initial_fitness)
               
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
                previous_fitness = self.get_current_best_fitness()
                self.ls_method = pool_global.get_new()
                self.apply_localsearch("global", evals_gs)
                improvement = self.get_ratio_improvement(previous_fitness, self.get_current_best_fitness())
                pool_global.improvement(self.ls_method, improvement, 2)
                self.set_best_global(self.current_best_id, self.current_best, self.current_best_fitness)
                self.log.info(f"algorithm={self.__class__.__name__} phase='Global Search' previous_fitness={previous_fitness} ls_method={self.ls_method} improvement={improvement} total_evals={self.totalevals} current_best={self.current_best_fitness} best_global={self.best_global_fitness}")

            self.set_region_ls()
            method = pool.get_new()

            if apply_de:
                self.compound_folder.append(f'turn_{g}')
                super().evolve()
                improvement = self.get_current_best_fitness() - self.best_fitness
                self.totalevals += len(self.G)
                self.set_current_best(self.best_id, self.best, self.best_fitness)
                self.log.info(f"algorithm={self.__class__.__name__} phase='DE' improvement={improvement} total_evals={self.totalevals} current_best={self.current_best_fitness} best_global={self.best_global_fitness}")

            if apply_ls:
                previous_fitness = self.get_current_best_fitness()
                self.apply_localsearch("local", evals_ls)
                improvement = self.get_ratio_improvement(previous_fitness, self.get_current_best_fitness())
                pool.improvement(method, improvement, 10, .25)
                self.log.info(f"algorithm={self.__class__.__name__} phase='Local Search' previous_fitness={previous_fitness} ls_method={method} improvement={improvement} total_evals={self.totalevals} current_best={self.current_best_fitness} best_global={self.best_global_fitness}")

            self.set_best_global(self.current_best_id, self.current_best, self.current_best_fitness)
            
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
  