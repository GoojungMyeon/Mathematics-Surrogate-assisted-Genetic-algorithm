import numpy as np
from random import uniform


def Selection_roulettewheel(population, fitness, selected_num, s_pressure):
    adj_fit = np.zeros(len(fitness))
    f_min = np.min(fitness)
    f_max = np.max(fitness)

    for fit_ind in range(len(fitness)):
        adj_fit[fit_ind] = f_min - fitness[fit_ind] + \
            (f_min-f_max)/(s_pressure-1)

    wheel = np.sum(adj_fit)
    selected_individuals = []
    while len(selected_individuals) != selected_num:
        pick = uniform(0, wheel) * 1
        s_sum = 0
        for ind_num in range(len(population)):
            s_sum += adj_fit[ind_num]
            if s_sum <= pick:
                selected_individuals.append(population[ind_num])
                break

    return selected_individuals
