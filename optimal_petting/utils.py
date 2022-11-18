import random 
import numpy as np


def random_attractiveness(option):
    return random.uniform(0, 1)


def get_sample_from_norm(mu=0.5, sigma=0.1, n_samples=100):
    return [random.normalvariate(mu,sigma) for i in range(n_samples)]


def get_relative_rank(x):
    """
    Gives relative rank of scores,
    and returns last rank if tied.
    e.g.,
        [0.5, 0.5, 0.2] -> [1,1,3]

    Note: 1 = top rank. 
    """
    rr = np.ones(len(x))
    x_ = np.array(x)

    for i in range(1,len(x)):
        n_values_better = np.sum(x_[i] < x_[:i+1])
    
        rr[i] = n_values_better + 1  

    return rr
