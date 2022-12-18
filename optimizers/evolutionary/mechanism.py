import asyncio
from random import random
import torch
import numpy as np
from .pytorch_utils import pytorch_model_set_weights_v2, pytorch_model_get_weights
from torch.nn.functional import cross_entropy
import copy
from multiprocessing import Pool
from functools import partial

async def evaluate_async(model, inputs, labels, individual):
    pytorch_model_set_weights_v2(model, individual)
    outputs = model(inputs)
    loss = cross_entropy(outputs, labels)
    return loss.item()

def evaluate(model, inputs, labels, individual):
    pytorch_model_set_weights_v2(model, individual)
    outputs = model(inputs)
    loss = cross_entropy(outputs, labels)
    return loss.item()

def evaluate_pop(model, inputs, labels, population):
	fitness_list = []

	for individual in population:
		fitness_list.append(evaluate(model, inputs, labels, individual))

	return fitness_list

def evaluate_pop_v2(model, inputs, labels, population):
	for individual in population:
		yield evaluate(model, inputs, labels, individual)
  
async def evaluate_pop_v3(model, inputs, labels, population):
    population_fitness = []
	
    for individual in population:
        population_fitness.append(await evaluate_async(model, inputs, labels, individual))
                                  
    return population_fitness

def create_population(size, model, device=None):
    population = []
    
    for _ in range(size):
        population.append(individual_v2(model, device))
  
    return population

def create_population_v2(size, model, device=None):
    for _ in range(size):
        yield individual_v2(model, device)

def individual(model, device=None):
	model_dict = model.state_dict()
	tensor_list = []

	for layer_name in model_dict:
		if 'weight' in layer_name:
			size = model_dict[layer_name].size()
			if device == None:
				tensor_list.append(torch.rand(size))
			else:
				tensor_list.append(torch.rand(size, device=device))

	return tensor_list

def individual_v2(model, device=None):
	parameters = model.parameters()
	tensor_list = []

	for param in parameters:
		size = param.data.size()
		if device == None:
			tensor_list.append(torch.rand(size))
		else:
			tensor_list.append(torch.rand(size, device=device))

	return tensor_list

async def individual_v3(model, device=None):
	parameters = model.parameters()
	tensor_list = []

	for param in parameters:
		size = param.data.size()
		if device == None:
			tensor_list.append(torch.rand(size))
		else:
			tensor_list.append(torch.rand(size, device=device))

	return tensor_list

def roulette_wheel_selection(population, fitness, size=None):
		fitness_sum = sum([f for f in fitness])
		probabilities = [f/fitness_sum for f in fitness]
		choices = np.random.choice(len(population), size=size, p=probabilities)
		return [population[chosen] for chosen in choices]

def sexual_inheritance_fitness(self, f, f1, f2):
		parent_fitness_mean = (f1 + f2) / 2
		return (parent_fitness_mean * (1 - self.decay)) + f

def asexual_inheritance_fitness(f, f1, decay):
		return (f1 * (1 - decay)) + f

def tournament_selection(self):
		pass

def offspring(model, inputs, labels, population, inheritance_fitness):
    fitness = []
    offspring = []
    for individual, old_fitness in zip(population, inheritance_fitness):
        (a, b, c) = roulette_wheel_selection(population, inheritance_fitness, size=3)
        mutant = mutation(copy.deepcopy(a), copy.deepcopy(b), copy.deepcopy(c), 0.15, (-1, 1))
        f = evaluate(model, inputs, labels, copy.deepcopy(mutant))
        if f < old_fitness:
            offspring.append(mutant)
            fitness.append(f)
        else:
            offspring.append(individual)
            fitness.append(old_fitness)
    return (offspring, fitness)

def offspring_v2(model, inputs, labels, population, fitness):
    best = population[np.argmin(fitness)]
    (a, b, c) = roulette_wheel_selection(population, fitness, size=3)
    mutant = mutation(a, b, c, 0.27, (-1, 1))
    offspring = [individual_v2(model) if np.random.random() < 0.5 else i for i in population]
    offspring[0] = mutant
    offspring[1] = best
    return (offspring, list(evaluate_pop_v2(model, inputs, labels, population)))

def offspring_v3(model, device, inputs, labels, population, fitness):
    best = population[np.argmin(fitness)]
    offspring_genereted_by_crossover = []
    offspring_genereted_by_mutation = []
    offspring_genereted_by_creation = []
    
    runs = int((len(population) - 1) / 3)
    
    for _ in range(runs):
        (a, b) = [population[i] for i in np.random.choice(len(population), size=2)]
        offspring_genereted_by_crossover.append(crossover(a, b, 0.58))
    
        (a, b, c) = roulette_wheel_selection(population, fitness, size=3)
        offspring_genereted_by_mutation.append(mutation(a, b, c, 0.27, (-1, 1)))

        offspring_genereted_by_creation.append(individual_v2(model, device))

    offspring = [best] + offspring_genereted_by_crossover + offspring_genereted_by_mutation + offspring_genereted_by_creation
    return (offspring, list(evaluate_pop_v2(model, inputs, labels, offspring)))

def offspring_v4(model, device, inputs, labels, population, fitness):
    best = population[np.argmin(fitness)]
    offspring_genereted_by_crossover = []
    offspring_genereted_by_mutation = []
    offspring_genereted_by_creation = []
    
    runs = int((len(population) - 1) / 3)
    
    for _ in range(runs):
        (a, b) = [population[i] for i in np.random.choice(len(population), size=2)]
        offspring_genereted_by_crossover.append(uniform_crossover_by_layer(a, b, 0.58))
    
        (index) = np.random.choice(len(population))
        offspring_genereted_by_mutation.append(random_reset_mutation(population[index]))

        offspring_genereted_by_creation.append(individual_v2(model, device))

    offspring = [best] + offspring_genereted_by_crossover + offspring_genereted_by_mutation + offspring_genereted_by_creation
    return (offspring, list(evaluate_pop_v2(model, inputs, labels, offspring)))

def mutation(a, b, c, F, bounds):
    if not (len(a) == len(b) and len(b) == len(c)):
        raise Exception("")
    
    individual_mutant = []
    
    for ai, bi, ci in zip(a, b, c):
        tensor = ai + F * (bi - ci)
        individual_mutant.append(torch.clamp(tensor, min=bounds[0], max=bounds[1]))
        
    return individual_mutant

def random_reset_mutation(a):
    mutant = copy.deepcopy(a)
    layer = np.random.randint(0, len(a))
    chromossomos = np.random.randint(0, len(a[layer]), size=int(len(a[layer]) *0.2))
    
    for c in chromossomos:
        mutant[layer][c] = np.random.random()
    
    return mutant
    
def crossover(i_A, i_B, Cr):
    mutant = []
    
    for tensor_weights_A, tensor_weights_B in zip(i_A, i_B):
        mutant.append(tensor_weights_A if random() < Cr else tensor_weights_B)
    
    return mutant

def uniform_crossover_by_layer(i_A, i_B, Cr):
    mutant = []
    
    for tensor_weights_A, tensor_weights_B in zip(i_A, i_B):
        mutant.append(tensor_weights_A if random() < Cr else tensor_weights_B)
    
    return mutant
