from src.nn_creature import Creature
from src.utils import *
import numpy as np
import random

from torch.multiprocessing import Pool, Queue, set_start_method
import torch
from pytorch_model_summary import summary
from examples.networks import MnistPytorch, PolygonOnCirclesPytorch, BiasedCowBunPytorch

processesGlobal = Queue()

def initModel(modelName):
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    global model
    global deviceUsed 
    deviceUsed = 'cpu' if processesGlobal.get() == '/device:CPU:0' else 'cuda'

    if modelName is 'MnistPytorch':
        model = MnistPytorch().to(torch.device(deviceUsed))
        summary(model, torch.zeros((1, 1, 28, 28)).to(torch.device(deviceUsed)), show_input=True, print_summary=True)
    elif modelName is 'PolygonOnCirclesPytorch':
        model = PolygonOnCirclesPytorch().to(torch.device(deviceUsed))
        summary(model, torch.zeros((1, 1, 50, 50)).to(torch.device(deviceUsed)), show_input=True, print_summary=True)
    elif modelName is 'BiasedCowBunPytorch':
        model = BiasedCowBunPytorch().to(torch.device(deviceUsed))
        summary(model, torch.zeros((1, 1, 50, 50)).to(torch.device(deviceUsed)), show_input=True, print_summary=True)


def createCreature(index):
	global model
	return Creature.fromPytorchModel(model)

def evalCreatures(data):
	creature = data['creature']
	batch_x = data['batch_x']
	batch_y = data['batch_y']

	global model
	global deviceUsed

	# global session
	# global graph
	# with session.as_default():
	# 	with graph.as_default():
	creature.evalPytorch(model, batch_x, batch_y, deviceUsed)
	return creature

def saveToDisk(args):
	# global session
	global model
	global deviceUsed
	# global graph
	# with session.as_default():
	# 	with graph.as_default(): 
	saveCreatureToDiskPytorch(args["generations"], args["generation"], args['creature'], args["mutationPower"], args["processedImages"], args["outputFolder"], model, deviceUsed)

class Neuroevolution:

	def __init__(self, processes, modelName, x_train, y_train, batchSize, numCreatures):

		for processDevice in processes:
			print(processDevice)
			processesGlobal.put(processDevice)

		self.processedImages = 0
		self.mutationPower = 0.01
		self.mutationProportion = 0.04
		self.crossoverProportion = 0.5
		self.x_train = x_train
		self.y_train = y_train
		self.batchSize = batchSize
		self.numCreatures = numCreatures

		self.x_train_training = copy.deepcopy(self.x_train)
		self.y_train_training = copy.deepcopy(self.y_train)

		# inicializa pool de redes keras
		self.executor = Pool(len(processes), initializer=initModel, initargs=(modelName,))

		# # inicializa população
		self.creatures = list(self.executor.map(createCreature, range(numCreatures)))
		
		# self.mutationPowerLayer = [self.mutationPower] *  self.creatures[0].numLayers()
		# self.mutationProportionLayer = [self.mutationProportion] * self.creatures[0].numLayers()

		self.generation = 0

		# SHADE VARIABLES
		self.shade_k = 0
		self.shade_H = 10
		self.shade_memoryCR = [0.5 for i in range(self.shade_H)]
		self.shade_memoryF = [0.5 for i in range(self.shade_H)]
		self.shade_externalArchiveA = []

		# DYNAMIC RAND 1 BIN VARIABLES
		self.rand_1_bin_cr = 0.5
		self.rand_1_bin_f = 0.5

	def loadNewGenerationBatch(self):

		[batch_x_train, batch_y_train] = miniBatch(self.x_train, self.y_train, self.batchSize)
		[self.batch_x_train, self.batch_y_train] = [np.array(batch_x_train), np.array(batch_y_train)]

	def loadNewGenerationBatchWithoutReplacement(self):

		assert len(self.x_train_training) == len(self.y_train_training)

		if (len(self.x_train_training) < self.batchSize * 10):

			self.x_train_training = copy.deepcopy(self.x_train)
			self.y_train_training = copy.deepcopy(self.y_train)

		def Diff(list1, list2): 
			return (list(list(set(list1)-set(list2)) + list(set(list2)-set(list1)))) 

		allIndexes = range(len(self.x_train_training))
		sampleIndexes = random.sample(allIndexes,self.batchSize*10)

		diffSamples = Diff(allIndexes, sampleIndexes)

		batch_x_train = self.x_train_training[sampleIndexes]
		batch_y_train = self.y_train_training[sampleIndexes]

		[self.batch_x_train, self.batch_y_train] = [np.array(batch_x_train), np.array(batch_y_train)]
		
		
		self.x_train_training = self.x_train_training[diffSamples]
		self.y_train_training = self.y_train_training[diffSamples]

	def evalPopulation(self):

		mapCreatureFunc = lambda x: { 'creature': x, 'batch_x': self.batch_x_train, 'batch_y':  self.batch_y_train}
		self.creatures = list(self.executor.map(evalCreatures, list(map(mapCreatureFunc, self.creatures))))

	def modifyPopulationFitness(self, fitnessInheritanceDecayRate):

		self.creatures = list(map(lambda x: modifyFitnessFunc(x, fitnessInheritanceDecayRate), self.creatures))

	def showBestGenerationNetworks(self, generation):
	
		self.creatures.sort(key=lambda x: x.fitness, reverse=True)
		for creature in self.creatures[:10]:
			creature.show(generation)
		random.shuffle(self.creatures)

	def updateMutationParams(self):

		[rate, proportion] = getUpdatedMutationRateProportion(self.mutationPower, self.mutationProportion, self.creatures)
		self.mutationPower = rate
		self.mutationProportion = proportion
		print('updated', self.mutationPower, self.mutationProportion)

	def updateMutationParamsByLayers(self):

		[rate, proportion] = getUpdatedMutationRateProportionLayers(self.mutationPowerLayer, self.mutationProportionLayer, self.creatures)
		self.mutationPowerLayer = rate
		self.mutationProportionLayer = proportion
		print('updated by layers', self.mutationPowerLayer, self.mutationProportionLayer)
	

	def getElitismNetworks(self, percentage):
		networksCount = int(len(self.creatures)*percentage)

		return self.getNElitismNetworks(networksCount)

	def getNElitismNetworks(self, n):
		elite = []
		self.creatures.sort(key=lambda x: x.fitness, reverse=True)
		generationBest = self.creatures[:n]
		for creature in generationBest:
			elite.append(creature.clone())
			
		for creature in elite:
			creature.parentAFitness = creature.fitness
			creature.generatedBy = 'Elitismo'
		random.shuffle(self.creatures)

		return elite

	def saveBestGenerationNetwork(self, generations, generation, outputFolder):

		self.creatures.sort(key=lambda x: x.fitness, reverse=True)
		list(self.executor.map(saveToDisk, [ {
		"generations": generations,
		"generation": generation,
		"mutationPower": self.mutationPower,
		"processedImages": self.processedImages,
		"creature": self.creatures[0],
		"outputFolder": outputFolder
		}]))
		random.shuffle(self.creatures)

	# def getMutationNetworksByLayer(self, proportion):

	# 	return list(map(lambda x: mutateCreatureFunc2(tournamentSelection(self.creatures), self.mutationPowerLayer, self.mutationProportionLayer), range(int(len(self.creatures) * proportion))))

	def getMutationNetworks(self, proportion):

		return list(map(lambda x: mutateCreatureFunc(tournamentSelection(self.creatures), self.mutationPower, self.mutationProportion), range(int(len(self.creatures) * proportion))))

	def mutateNetwork(self, network):
		return mutateCreatureFunc(network, self.mutationPower, self.mutationProportion)

	def mutateNetworkNtimes(self, network, n):
		return list(map(lambda x: mutateCreatureFunc(network, self.mutationPower, self.mutationProportion), [network]*n ))

	def getCrossoverNetworks(self, proportion):

		return list(map(lambda x: crossoverCreatureFunc(tournamentSelection(self.creatures), tournamentSelection(self.creatures)), range(int(len(self.creatures) * proportion))))

	# def getDifferentialBest1BinNetworks(self):
		
	# 	trialsVectors = []
	# 	for parentVector in self.creatures:
			
	# 		populationWithoutParent = difference(self.creatures, [parentVector])

	# 		# best 1 bin
	# 		populationWithoutParent.sort(key=lambda x: x.fitness, reverse=True)
	# 		targetVector = populationWithoutParent[0]

	# 		populationWithoutParentAndTarget = difference(self.creatures, [parentVector, targetVector])

	# 		[x1, x2] = random.sample(populationWithoutParentAndTarget, 2)
	# 		# trialVector = targetVector + ( x1 - x2 ) * 0.5
	# 		trialVector = targetVector.sumWith(x1.subtratedWith(x2).multipliedByNatural(0.5))
			
	# 		# 0.7 ou 1.0
	# 		[trialVectorCrossed, y] = parentVector.crossedWith(trialVector, 0.7)
	# 		trialVectorCrossed.clamp(-1.0, 1.0)
	# 		# trialVectorCrossed.clamp_rand(-1.0, 1.0)
			

	# 		trialsVectors.append(trialVectorCrossed)

	# 	mapCreatureFunc = lambda x: { 'creature': x, 'batch_x': self.batch_x_train, 'batch_y':  self.batch_y_train}
	# 	trialsVectorsEvaluated = list(self.executor.map(evalCreatures, list(map(mapCreatureFunc, trialsVectors))))


	# 	bestCreatures = []
	# 	for parent, trial in zip(self.creatures, trialsVectorsEvaluated):

	# 		if parent.fitness > trial.fitness:
	# 			bestCreatures.append(parent)
	# 		else:
	# 			bestCreatures.append(trial)

	# 	return bestCreatures

	# def getDifferentialBest1BinNetworksAdaptativeF_CR(self):
		
	# 	trialsVectors = []
	# 	trialsCR_used = []
	# 	trialsF_used = []
	# 	for parentVector in self.creatures:
			
	# 		populationWithoutParent = difference(self.creatures, [parentVector])

	# 		# best 1 bin
	# 		populationWithoutParent.sort(key=lambda x: x.fitness, reverse=True)
	# 		targetVector = populationWithoutParent[0]

	# 		populationWithoutParentAndTarget = difference(self.creatures, [parentVector, targetVector])

	# 		[x1, x2] = random.sample(populationWithoutParentAndTarget, 2)

	# 		cr = max([0, min([self.rand_1_bin_cr * 1.1 if random.random() > 0.5 else self.rand_1_bin_cr * 0.9,1])])  
	# 		f = max([0, min([self.rand_1_bin_f * 1.1 if random.random() > 0.5 else self.rand_1_bin_f * 0.9,1])])  

	# 		# trialVector = targetVector + ( x1 - x2 ) * 0.5
	# 		trialVector = targetVector.sumWith(x1.subtratedWith(x2).multipliedByNatural(f))
			
	# 		trialsCR_used.append(cr)
	# 		trialsF_used.append(f)

	# 		# 0.7 ou 1.0
	# 		[trialVectorCrossed, y] = parentVector.crossedWith(trialVector, cr)
	# 		trialVectorCrossed.clamp(-1.0, 1.0)

	# 		trialsVectors.append(trialVectorCrossed)

	# 	mapCreatureFunc = lambda x: { 'creature': x, 'batch_x': self.batch_x_train, 'batch_y':  self.batch_y_train}
	# 	trialsVectorsEvaluated = list(self.executor.map(evalCreatures, list(map(mapCreatureFunc, trialsVectors))))


	# 	bestCreatures = []
	# 	success_f = []
	# 	success_cr = []
	# 	# for parent, trial in zip(self.creatures, trialsVectorsEvaluated):
	# 	for parent, trial, usedCr, usedF in zip(self.creatures, trialsVectorsEvaluated, trialsCR_used, trialsF_used):
	# 		if parent.fitness > trial.fitness:
	# 			bestCreatures.append(parent)
	# 		else:
	# 			bestCreatures.append(trial)
	# 			success_f.append(usedF)
	# 			success_cr.append(usedCr)
		
	# 	success_f.append(self.rand_1_bin_f)
	# 	if len(success_f) > 0:
	# 		mean = sum(success_f) / len(success_f)
	# 		self.rand_1_bin_f = mean

	# 	success_cr.append(self.rand_1_bin_cr)
	# 	if len(success_cr) > 0:
	# 		mean = sum(success_cr) / len(success_cr)
	# 		self.rand_1_bin_cr = mean

	# 	print("Updated CR:", self.rand_1_bin_cr, "Updated F:", self.rand_1_bin_f)


	# 	return bestCreatures

	# def getDifferentialBestWorst1BinNetworks(self):
		
	# 	trialsVectors = []
	# 	for parentVector in self.creatures:
			
	# 		populationWithoutParent = difference(self.creatures, [parentVector])

	# 		# best
	# 		populationWithoutParent.sort(key=lambda x: x.fitness, reverse=True)
	# 		bestVector = populationWithoutParent[0]
	# 		worstVector = populationWithoutParent[-1]


	# 		populationWithoutParentAndTarget = difference(self.creatures, [parentVector, bestVector, worstVector])

	# 		[x1, x2, x3] = random.sample(populationWithoutParentAndTarget, 3)
	# 		# trialVector = x1 + f1 * (best - x2) + f1 * (x3 - worst)
	# 		f1 = random.random()
	# 		f2 = random.random()
	# 		trialVector = x1.sumWith(bestVector.subtratedWith(x2).multipliedByNatural(f1)).sumWith(x3.subtratedWith(worstVector).multipliedByNatural(f2))
			
	# 		# 0.7 ou 1.0
	# 		[trialVectorCrossed, y] = parentVector.crossedWith(trialVector, 0.5)
	# 		trialVectorCrossed.clamp(-1, 1)
			
	# 		trialsVectors.append(trialVectorCrossed)

	# 	mapCreatureFunc = lambda x: { 'creature': x, 'batch_x': self.batch_x_train, 'batch_y':  self.batch_y_train}
	# 	trialsVectorsEvaluated = list(self.executor.map(evalCreatures, list(map(mapCreatureFunc, trialsVectors))))


	# 	bestCreatures = []
	# 	for parent, trial in zip(self.creatures, trialsVectorsEvaluated):

	# 		if parent.fitness > trial.fitness:
	# 			bestCreatures.append(parent)
	# 		else:
	# 			bestCreatures.append(trial)

	# 	return bestCreatures


	# def getDifferentialNetworks(self):
		
	# 	trialsVectors = []
	# 	for parentVector in self.creatures:
			
	# 		populationWithoutParent = difference(self.creatures, [parentVector])

	# 		[targetVector, x1, x2] = random.sample(populationWithoutParent, 3)
	# 		# trialVector = targetVector + ( x1 - x2 ) * 0.5
	# 		trialVector = targetVector.sumWith(x1.subtratedWith(x2).multipliedByNatural(0.5))
			

	# 		[trialVectorCrossed, y] = parentVector.crossedWith(trialVector)
	# 		trialVectorCrossed.clamp(-1.0, 1.0)
			
	# 		trialsVectors.append(trialVectorCrossed)


	# 	mapCreatureFunc = lambda x: { 'creature': x, 'batch_x': self.batch_x_train, 'batch_y':  self.batch_y_train}
	# 	trialsVectorsEvaluated = list(self.executor.map(evalCreatures, list(map(mapCreatureFunc, trialsVectors))))

	# 	bestCreatures = []
	# 	for parent, trial in zip(self.creatures, trialsVectorsEvaluated):

	# 		if parent.fitness > trial.fitness:
	# 			bestCreatures.append(parent)
	# 		else:
	# 			bestCreatures.append(trial)

	# 	return bestCreatures

	# def getDifferentialNetworksAdaptativeF_CR(self):
		
	# 	trialsVectors = []
	# 	trialsCR_used = []
	# 	trialsF_used = []
	# 	for parentVector in self.creatures:
			
	# 		populationWithoutParent = difference(self.creatures, [parentVector])

	# 		cr = max([0, min([self.rand_1_bin_cr * 1.1 if random.random() > 0.5 else self.rand_1_bin_cr * 0.9,1])])  
	# 		f = max([0, min([self.rand_1_bin_f * 1.1 if random.random() > 0.5 else self.rand_1_bin_f * 0.9,1])])  

	# 		[targetVector, x1, x2] = random.sample(populationWithoutParent, 3)
	# 		# trialVector = targetVector + ( x1 - x2 ) * 0.5
	# 		trialVector = targetVector.sumWith(x1.subtratedWith(x2).multipliedByNatural(f))
			
			
	# 		[trialVectorCrossed, y] = parentVector.crossedWith(trialVector, cr)
	# 		trialVectorCrossed.clamp(-1.0, 1.0)

	# 		trialsCR_used.append(cr)
	# 		trialsF_used.append(f)

	# 		trialVectorCrossed.mutationPowerUsed = self.mutationPower
	# 		trialVectorCrossed.crossoverProportionUsed = self.crossoverProportion

	# 		trialsVectors.append(trialVectorCrossed)


	# 	mapCreatureFunc = lambda x: { 'creature': x, 'batch_x': self.batch_x_train, 'batch_y':  self.batch_y_train}
	# 	trialsVectorsEvaluated = list(self.executor.map(evalCreatures, list(map(mapCreatureFunc, trialsVectors))))

	# 	bestCreatures = []
	# 	success_f = []
	# 	success_cr = []
	# 	for parent, trial, usedCr, usedF in zip(self.creatures, trialsVectorsEvaluated, trialsCR_used, trialsF_used):

	# 		if parent.fitness > trial.fitness:
	# 			bestCreatures.append(parent)
	# 		else:
	# 			bestCreatures.append(trial)
	# 			success_f.append(usedF)
	# 			success_cr.append(usedCr)
		
	# 	success_f.append(self.rand_1_bin_f)
	# 	if len(success_f) > 0:
	# 		mean = sum(success_f) / len(success_f)
	# 		self.rand_1_bin_f = mean

	# 	success_cr.append(self.rand_1_bin_cr)
	# 	if len(success_cr) > 0:
	# 		mean = sum(success_cr) / len(success_cr)
	# 		self.rand_1_bin_cr = mean

	# 	print("Updated CR:", self.rand_1_bin_cr, "Updated F:", self.rand_1_bin_f)

	# 	return bestCreatures

	# def getDifferentialSHADENetworks(self):
	# 	#Article
	# 	#Success History-Based Adaptive Differential Evolution Using Turning-Based Mutation
		
	# 	# Initialization
	# 	trialsVectors = []
	# 	trialVectorsF = []
	# 	trialVectorsCR = []
	# 	for parentVector in self.creatures:

	# 		# Mutation
	# 		# trialVector = atual + F * (randBestX - atual) + F * (randPop - randPopArchive)
	# 		pMin = 2 / self.numCreatures
	# 		pi = random.uniform(pMin, 0.2)
			
	# 		populationWithoutParent = difference(self.creatures, [parentVector])

	# 		# best
	# 		populationWithoutParent.sort(key=lambda x: x.fitness, reverse=True)
	# 		nBestToGet = self.numCreatures * pi

	# 		bestNVector = populationWithoutParent[:int(nBestToGet)]
	# 		bestVector = random.choice(bestNVector)

	# 		# TODO
	# 		#cauchyDistribution = lambda x,y: 1/(math.pi*(1+(math.pow(x-y, 2))))
			
	# 		fi = -1
	# 		while fi < 0:
	# 			randomF = random.choice(self.shade_memoryF)
	# 			fi = random.uniform(randomF, 0.1)
	# 			if fi > 1:
	# 				fi = 1

	# 		populationWithoutParentAndTarget = difference(self.creatures, [parentVector, bestVector])
	# 		[popRand] = random.sample(populationWithoutParentAndTarget, 1)

	# 		populationWithoutParentTargetAndPopRand = difference(self.creatures, [parentVector, bestVector, popRand])

	# 		[popArchiveRand] = random.sample(populationWithoutParentTargetAndPopRand + self.shade_externalArchiveA, 1)

	# 		trialVector = parentVector.sumWith(bestVector.subtratedWith(parentVector).multipliedByNatural(fi)).sumWith(popRand.subtratedWith(popArchiveRand).multipliedByNatural(fi))
	# 		# boundary handling ?
	# 		trialVectorCrossed.clamp(-1.0, 1.0)

	# 		# Crossover
	# 		randomCR = random.choice(self.shade_memoryCR)
	# 		randomCR = max([0, min([randomCR,1])])
	# 		cri = np.random.normal(randomCR,0.1)
	# 		cri = max([0, min([cri,1])])
	# 		[trialVectorCrossed, y] = parentVector.crossedWith(trialVector, cri)

	# 		# Selection
	# 		trialsVectors.append(trialVectorCrossed)
	# 		trialVectorsF.append(fi)
	# 		trialVectorsCR.append(cri)

	# 	mapCreatureFunc = lambda x: { 'creature': x, 'batch_x': self.batch_x_train, 'batch_y':  self.batch_y_train}
	# 	trialsVectorsEvaluated = list(self.executor.map(evalCreatures, list(map(mapCreatureFunc, trialsVectors))))

	# 	bestCreatures = []
	# 	success_f = []
	# 	success_cr = []
	# 	failed_parents = []
	# 	success_trials = []
	# 	for parent, trial, fi_used, cri_used in zip(self.creatures, trialsVectorsEvaluated, trialVectorsF, trialVectorsCR):

	# 		if parent.fitness > trial.fitness:
	# 			bestCreatures.append(parent)
	# 		else:
	# 			# store parent on A
	# 			self.shade_externalArchiveA += [parent]
	# 			self.shade_externalArchiveA = random.sample(self.shade_externalArchiveA, min(len(self.shade_externalArchiveA), 10))
	# 			# add success
	# 			success_f.append(fi_used)
	# 			success_cr.append(cri_used)
	# 			parent.fitness > trial.fitness
	# 			failed_parents.append(parent.loss)
	# 			success_trials.append(trial.loss)

	# 			bestCreatures.append(trial)

	# 	delta_fk = lambda index: abs(success_trials[index] - failed_parents[index])
	# 	w = lambda k: delta_fk(k) / sum([delta_fk(z) for z in range(len(success_cr))])

	# 	if len(success_cr) > 0:
	# 		mean_cr = sum([w(i) * success_cr[i] for i in range(len(success_cr))])
	# 		self.shade_memoryCR[self.shade_k % self.shade_H] = mean_cr
			
	# 		dividendo = sum([w(i) * success_f[i] for i in range(len(success_cr))])
	# 		if dividendo > 0:
	# 			mean_f = sum([w(i) * pow(success_f[i],2) for i in range(len(success_cr))]) / dividendo
	# 			self.shade_memoryF[self.shade_k % self.shade_H] = mean_f

	# 	self.shade_k += 1

	# 	print(self.shade_k, self.shade_externalArchiveA, self.shade_memoryCR, self.shade_memoryF)

	# 	return bestCreatures

	def replacePopulation(self, newNetworks):

		self.creatures = copy.deepcopy(newNetworks)

		assert len(self.creatures) == self.numCreatures


	def destroyPopulation(self):

		self.executor.terminate()