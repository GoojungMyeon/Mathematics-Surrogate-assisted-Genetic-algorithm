import torch
import time
import os
import numpy as np
import pandas as pd
import joblib
import sys


import GA.crossover
import GA.evaluate
import GA.init_pops
import GA.mutation
import GA.selection

import Surrogates.Classifier.DNN_Classifiers

import config

architecture_set = ["1_3", "1_5", "2_3_3", "2_5_5", "2_32_32"]

if __name__ == "__main__":

    assert (config.Environemnt == "Cartpole" or config.Environemnt ==
            "Lunar"), "Wrong Environment"
    assert (config.Architecture in architecture_set), "Wrong Architecture"
    assert (config.penalty1 > 0 and config.penalty2 >
            0), "Wrong penalty weight"

    Chr_length = 0


    assert Chr_length != 0 , "Wrong Architecture"
    regressor = joblib.load("./Surrogates/Regressor/"+config.Environemnt+"/SVR_"+config.Architecture+".pkl")
    classifier_1 = Surrogates.Classifier.DNN_Classifiers(Chr_length)
    classifier_1.load_state_dict("Surrogates/Classifier/"+config.Environemnt+"/Clf1_"+config.Architecture+".pt")
    classifier_2 = Surrogates.Classifier.DNN_Classifiers(Chr_length)
    classifier_2.load_state_dict("Surrogates/Classifier/"+config.Environemnt+"/Clf2_"+config.Architecture+".pt")
    B_set = pd.read_csv("Surrogates/Boundary/"+config.Environemnt+"/Boundary_"+config.Architecture+".csv")

    pops = GA.init_pops.generate_population_np(config.init_size,Chr_length,B_set)
    pop_id = np.zeros(config.pop_size)
    fitness = []

    fitness = GA.evaluate.Fit_evaluate(pops,classifier_1,classifier_2,regressor,config.penalty1,config.penalty2)
    pd.DataFrame(pops).to_csv("./results/init_pops.csv")
    pd.DataFrame(fitness).to_csv("./results/init_fitnesses.csv")

    for gen in range(config.generation):
        selected_inds = GA.selection.Selection_roulettewheel(pops,fitness,config.selection_num,config.selection_num)
        next_inds = []
        for inds in range(int(len(selected_inds) / 2)):
            next_inds.append(GA.crossover.Crossover_ExtendedBox_Repair(selected_inds[inds*2],selected_inds[inds*2+1],config.Xover_alpha,B_set))

        for inds in range(len(next_inds)):
            next_inds[inds] = GA.mutation.Mutation(next_inds[inds],config.mu_prob,B_set)

        next_fitness = GA.evaluate.Fit_evaluate(
            next_inds, classifier_1, classifier_2, regressor, config.penalty1, config.penalty2)

        pops = np.concatenate([pops,next_inds])
        fitness = np.concatenate([fitness,next_fitness])

        pops = pops[fitness.argsort()[::-1]][0:config.pop_size]
        fitness = np.sort(fitness)[0:config.pop_size]


        pd.DataFrame(pops).to_csv("./results/"+"{0:04}".format(gen)+"pops.csv")
        pd.DataFrame(fitness).to_csv("./results/"+"{0:04}".format(gen)+"fitnesses.csv")
