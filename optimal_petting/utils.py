import random 
import numpy as np
from scipy.stats import rv_continuous
from typing import List, Tuple

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


def weight_expected_value(expected_value, dist):
    return expected_value * dist.cdf(expected_value)


def get_optimal_thresholds(
    n_options: int, 
    dist: rv_continuous
    ) -> Tuple[List[float], List[float]]:
    """ 
    Calculates the optimal stopping thresholds via backward
    induction, replicating the methodology used in:

      A linear threshold model for optimal stopping behavior
        https://doi.org/10.1073/pnas.2002312117

    - The threshold of the final item is ∞, because the
      rules of the task stipulate that the final item must be
      accepted if no earlier item has been chosen.
       
    - The thresholds for the previous items are determined by
      working backward from the final item, using conditional
      expectations.

    Important Assumptions:
    - Lower values are better/more rewarding (e.g., Price)

    Params:
        n_options (Int): number of observations to choose from 
                        (e.g., candidates).
        dist (scipy.stats.rv_continuous): the distribution the
                        options were sampled from.
    Returns: thresholds, expected_values
    """
    # thresh = the acceptance threshold, expected = expected value
    thresh, expected = np.zeros(n_options), np.zeros(n_options)

    # Treat the start of the list as the last option seen (reverse list at end).
    thresh[0] = np.inf
    expected[0] = dist.mean()
    thresh[1] = expected[0]

    for i in range(1, n_options-1):
        # expected value of the area under the probabilty curve up until the threshold
        expected[i] = dist.expect(lb=-np.inf, ub=thresh[i])
        
        # probability of the expected value 
        prob = dist.cdf(thresh[i]) 
        
        thresh[i+1] = (expected[i] * prob) + ( # expected value of item * it's probability
                expected[i-1] * (1-prob) # the expected value of previous item * its probability
                )        

    # Fill in the remaining expectation value
    expected[-1] = dist.expect(lb=-np.inf, ub=thresh[-1])

    return thresh[::-1], expected[::-1]
