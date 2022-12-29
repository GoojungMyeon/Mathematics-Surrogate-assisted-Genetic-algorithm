import numpy as np
from random import uniform

def Crossover_ExtendedBox(individual_1, individual_2, extension_rate):
    offspring = np.zeros(len(individual_1))
    for dim in range(len(individual_1)):
        d_max = max(individual_1[dim], individual_2[dim])
        d_min = min(individual_1[dim], individual_2[dim])

        min_bound = d_min - (d_max - d_min) * extension_rate
        max_bound = d_max + (d_max - d_min) * extension_rate
        offspring[dim] = uniform(min_bound, max_bound)
    return offspring


def Crossover_ExtendedBox_Repair(individual_1, individual_2, extension_rate, B_set):
    offspring = np.zeros(len(individual_1))
    for dim in range(len(individual_1)):
        d_max = max(individual_1[dim], individual_2[dim])
        d_min = min(individual_1[dim], individual_2[dim])

        min_bound = d_min - (d_max - d_min) * extension_rate if d_min - (
            d_max - d_min) * extension_rate >= B_set["min"][dim] else B_set["min"][dim]
        max_bound = d_max + (d_max - d_min) * extension_rate if d_max + (
            d_max - d_min) * extension_rate <= B_set["max"][dim] else B_set["max"][dim]
        offspring[dim] = uniform(min_bound, max_bound)
    return offspring
