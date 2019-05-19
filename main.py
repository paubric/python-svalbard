import array
import random

import numpy as np

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

target = np.random.rand(100)

creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array, typecode='f',
               fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("attr_bool", random.random)

# Structure initializers
toolbox.register("individual", tools.initRepeat,
                 creator.Individual, toolbox.attr_bool, 100)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def genOffering(individual):
    offering = np.zeros((100))

    for i in range(len(individual)):
        random.seed(i)
        contribution = np.random.rand(100) * individual[i]
        offering += contribution

    offering = np.reshape(offering, (1, 100)).reshape(-1, 1)
    scaler = MinMaxScaler()
    scaler = scaler.fit(offering)
    offering = scaler.transform(offering)

    return [value[0] for value in offering]


def evalOneMax(individual):
    offering = genOffering(individual)

    return np.mean(np.square(target - offering)),


toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)


def main():
    random.seed(64)

    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.5, ngen=2000,
                                   stats=stats, halloffame=hof, verbose=True)

    offering = genOffering(hof[0])
    plt.bar(range(100), target)
    plt.bar(range(100), offering)
    plt.show()

    return pop, log, hof


if __name__ == "__main__":
    main()
