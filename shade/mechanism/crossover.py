from numpy.random import permutation, rand
from numpy import concatenate
from .pool import PoolInc

class Crossover(object):
    """
    This class wrap a simple crossover function with empties methods required for DE
    """
    def initrun(self, run, bounds, maxEvals, PS):
        """There is no code at the beginning of each run"""
        pass

    def apply(self, population, i, indexBest, F):
        """
        Applies the crossover function

        :population: from which apply the crossover
        :i: current position.
        :indexBest: index of best solution.
        :F: parameter F
        """
        pass

    def stats(self):
        """There is special statistics"""
        return ""

    def set_previous_improvement(self, account):
        return
    
class UniformCrossover(Crossover):
    """
    This class wrap a simple crossover function with empties methods required for DE
    """
    def initrun(self, run, bounds, maxEvals, PS):
        """There is no code at the beginning of each run"""
        pass

    def apply(self, base, trial, Cr):
        pass

class SADECrossover(Crossover):
    def __init__(self, LP=50):
        crossovers = [classicalBinFunction, classicalTwoBinFunction, classicalBestFunction, currentToRand]
        self.pool = PoolInc(crossovers)
        self.LP = LP
        self.PS = 0
        self.count_calls = 0

    def initrun(self, run, bounds, maxEvals, PS):
        self.PS = PS
        self.count_calls = 0
        self.gene = 0

    def apply(self, population, i, bestIndex, F):
        crossover = self.pool.get_new()
        self.last_crossover = crossover
        return crossover(population, i, bestIndex, F)

    def stats(self):
        cumprob = self.pool.get_prob()
        prob = cumprob - concatenate(([0], cumprob[0:-1]))
        return ' '.join(map(str, prob))
    
    def set_previous_improvement(self, improvement):
        """Update the pool command"""
        self.pool.improvement(self.last_crossover, improvement)
        self.count_calls += 1

        if self.count_calls == self.PS:
            self.count_calls = 0
            self.gene += 1

            if self.gene >= self.LP:
               self.pool.update_prob()
               
def classicalBinFunction(population, i, bestIndex, F):
    """
    Implements the classical crossover function for DE
    """
    (c, a, b) = permutation(len(population))[:3]
    noisyVector = population[c] + F * (population[a] - population[b])
    return noisyVector

def classicalTwoBinFunction(population, i, bestIndex, F):
    """
    Implements the classical crossover function for DE
    :param population: population
    :param i: current
    :param bestIndex: best global
    :param F: parameter
    """
    size = population.shape[0]

    (c, a, b, r3, r4) = permutation(size)[:5]
    noisyVector = population[c] + F * (population[a] - population[b])  + F * (population[r3] - population[r4])
    return noisyVector

def currentToRand(population, i, bestIndex, F):
    """
    Crossover with the DE/current-to-rand/1
    :param population: of solution
    :param i: current solution
    :param bestIndex: best current solution
    :param F: parameter
    :return: vector results
    """
    size = len(population)
    (r1, r2, r3) = permutation(size)[:3]
    k = rand()
    noisyVector = population[i]+k*(population[r1]-population[i])\
                               +F*(population[r2]-population[r3])

    return noisyVector


def classicalBestFunction(population, i, bestIndex, F):
    """
    Implements the classical DE/best/ mutation
    """
    (a, b) = permutation(len(population))[:2]
    noisyVector = population[bestIndex] + F * (population[a] - population[b])
    return noisyVector

def randToBestFunction(population, i, bestIndex, F):
    """
    Implements the DE/rand-to-best/2/bin

    :param population: of solutions
    :param i: iteration
    :param bestIndex: index of current best
    :param F: parameter F (ratio)
    :return: A vector with
    """
    size = len(population)
    (r1, r2, r3, r4) = permutation(size)[:4]
    noisy_vector = population[i]+F*(population[bestIndex]-population[i])\
                               +F*(population[r1]-population[r2])\
                               +F*(population[r3]-population[r4])
    return noisy_vector
