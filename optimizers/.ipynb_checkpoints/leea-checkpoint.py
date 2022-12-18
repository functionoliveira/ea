import time
# class that handles population evolution
from neuroevolution import Neuroevolution

def leea_train(generations, devices, modelName, x_train, y_train, batchSize, numCreatures, outputFolder):

    print("devices", devices)
    print("modelName", modelName)

    population = Neuroevolution(devices, modelName, x_train, y_train, batchSize, numCreatures)

    for generation in range(generations):
        startGeneration = time.time()

        start = time.time()
        population.loadNewGenerationBatchWithoutReplacement()
        # population.loadNewGenerationBatch()
        timeBatchSelect = time.time() - start

        start = time.time()
        population.evalPopulation()
        timeEval = time.time() - start

        start = time.time()
        # population.modifyPopulationFitness(0.9)
        # population.updateMutationParams() na real isso daqui é um GA com batch grande
        updateParams = time.time() - start

        population.showBestGenerationNetworks(generation)
        population.saveBestGenerationNetwork(generations, generation, outputFolder)

        start = time.time()
        elitism = population.getElitismNetworks(0.05)
        eliteTime = time.time() - start

        start = time.time()
        mutated = population.getMutationNetworks(0.45)
        mutationTime = time.time() - start

        start = time.time()
        crossed = population.getCrossoverNetworks(0.5)
        crossoverTime = time.time() - start

        # print("elitism + mutated + crossed", len(elitism), len(mutated), len(crossed))
        population.replacePopulation(elitism + mutated + crossed)
        generationTime = time.time() - startGeneration

        print('Select batch on:', timeBatchSelect, 'Evaluate on: ', timeEval, 'Update params on:', updateParams, 'Elitism on:', eliteTime, 'Mutation on:', mutationTime, 'Crossover on:', crossoverTime, 'Total:', generationTime)

    population.destroyPopulation()

def leea_adapt_train(generations, devices, modelName, x_train, y_train, batchSize, numCreatures, outputFolder):

    print("devices", devices)
    print("modelName", modelName)

    population = Neuroevolution(devices, modelName, x_train, y_train, batchSize, numCreatures)

    for generation in range(generations):
        startGeneration = time.time()

        start = time.time()
        population.loadNewGenerationBatchWithoutReplacement()
        # population.loadNewGenerationBatch()
        timeBatchSelect = time.time() - start

        start = time.time()
        population.evalPopulation()
        timeEval = time.time() - start

        start = time.time()
        # population.modifyPopulationFitness(0.9)
        population.updateMutationParams()
        updateParams = time.time() - start

        population.showBestGenerationNetworks(generation)
        population.saveBestGenerationNetwork(generations, generation, outputFolder)

        start = time.time()
        elitism = population.getElitismNetworks(0.05)
        eliteTime = time.time() - start

        start = time.time()
        mutated = population.getMutationNetworks(0.45)
        mutationTime = time.time() - start

        start = time.time()
        crossed = population.getCrossoverNetworks(0.5)
        crossoverTime = time.time() - start

        # print("elitism + mutated + crossed", len(elitism), len(mutated), len(crossed))
        population.replacePopulation(elitism + mutated + crossed)

        generationTime = time.time() - startGeneration

        print('Select batch on:', timeBatchSelect, 'Evaluate on: ', timeEval, 'Update params on:', updateParams, 'Elitism on:', eliteTime, 'Mutation on:', mutationTime, 'Crossover on:', crossoverTime, 'Total:', generationTime)

    population.destroyPopulation()
