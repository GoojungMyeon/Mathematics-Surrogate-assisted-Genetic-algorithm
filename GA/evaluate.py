import torch
import numpy as np

def Fit_evaluate(population,clf1,clf2,reg,pc1,pc2):
    new_fit = np.zeros(len(population))


    for chr_ind in range(len(population)):
        with torch.no_grad():
            if torch.round(torch.sigmoid(clf1(torch.tensor(population[chr_ind]).float()))) == 1:
                if torch.round(torch.sigmoid(clf2(torch.tensor(population[chr_ind]).float()))) == 1:
                    fit = reg.predict(population[chr_ind].reshape(1, -1))[0]
                    new_fit[chr_ind]
                else:
                    fit = reg.predict(population[chr_ind].reshape(1, -1))[0]
                    fit = pc2
                    new_fit[chr_ind]
            else:
                fit = reg.predict(population[chr_ind].reshape(1, -1))[0]
                fit = pc1
                new_fit[chr_ind]

    return new_fit
