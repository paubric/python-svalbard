import array
import random
import numpy as np
from deap import algorithms, base, creator, tools
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

length_of_signal = 10
number_of_seeds = 10
generations = 1000
population_size = 100
cxpb = 0.5
mutpb = 0.2

target = np.random.rand(length_of_signal)


def genOffering(individual):
    offering = np.zeros((length_of_signal))

    for i in range(len(individual)):
        random.seed(i)
        contribution = np.random.rand(length_of_signal) * individual[i]
        offering += contribution

    offering = np.reshape(offering, (1, length_of_signal)).reshape(-1, 1)
    scaler = MinMaxScaler()
    scaler = scaler.fit(offering)
    offering = scaler.transform(offering)

    return [value[0] for value in offering]


def evalOneMax(individual):
    offering = genOffering(individual)

    return np.mean(np.abs(target - offering)),


def main():
    creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
    creator.create("Individual", array.array, typecode='f',
                   fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # Attribute generator
    toolbox.register("attr_bool", random.random)

    # Structure initializers
    toolbox.register("individual", tools.initRepeat,
                     creator.Individual, toolbox.attr_bool, number_of_seeds)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evalOneMax)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=10)

    random.seed(64)

    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=generations,
                                   stats=stats, halloffame=hof, verbose=True)

    offering = genOffering(hof[0])
    plt.bar(range(length_of_signal), target - offering)
    plt.show()

    return pop, log, hof


if __name__ == "__main__":
    main()
