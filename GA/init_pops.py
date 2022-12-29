import numpy as np


def generate_population_np(size, chr_length, B_set):
    sum = np.zeros((size, 1))
    for gene_dim in range(chr_length):
        col = np.random.rand(
            size)*(B_set["max"][gene_dim]-B_set["min"][gene_dim])
        sum = np.concatenate([sum, np.reshape(col, (size, 1))], axis=1)
    sum = np.delete(sum, 0, axis=1)
    return sum
