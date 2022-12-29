import numpy as np
from random import uniform

def Mutation(chromosome, mu_prob, B_set):

    for gene_ind in range(len(chromosome)):
        mu_rand = uniform(0, 1)
        if mu_rand < mu_prob:
            chromosome[gene_ind] = uniform(
                B_set["min"][gene_ind], B_set["max"][gene_ind])
    return chromosome,
