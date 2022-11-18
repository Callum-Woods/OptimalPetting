import numpy as np
from optimal_petting.utils import get_relative_rank
from scipy.stats.distributions import norm


class IdealObserver:

    def __init__(
        self,
        expected_n_applicants: int,
        cost_to_sample: float,
        past_observations: list,
        reward_threshold: float,
        ):
        self.num_applicants = expected_n_applicants
        self.cost_to_sample = cost_to_sample
        self.past_observations = past_observations
        self.reward_threshold = reward_threshold
        
        self.historic_scores = []

        # TODO: Is this correct?
        self.prior = norm(
            np.mean(self.past_observations),
            np.std(self.past_observations)
        )
    
    def threshold_reward(self, option):
        if option < self.reward_threshold:
            reward = option
        else:
            reward = 0
        
        return reward

    def get_reward(self, option):
        # TODO: Allow expansion and provision of custom reward funcs
        return self.threshold_reward(option)

    def get_relative_rank_of_current_option(self, option):
        self.historic_scores.append(option)
        rr = get_relative_rank(self.historic_scores)

        return rr[-1]

    def prob_best_option(self, option):
        # TODO: This is not likelihood of top rank, but 
        # probability of seeing a value equal or less than this option 
        return self.prior.cdf(option)

    def get_likelihood_of_being_top_rank(relative_rank):
        # TODO: Implement
        pass

    def compute_take_value(self, option):
        # TODO: Check generic logic is correct

        # get_relative_rank has side effect: updates self.historic scores
        relative_rank = self.get_relative_rank_of_current_option(option)
        likelihood_of_being_best_option = self.prob_best_option(option)
        n_options_seen = len(self.historic_scores) 
        reward = self.get_reward(n_options_seen + (relative_rank - 1))

        return likelihood_of_being_best_option * reward

    def compute_decline_value(self, option):
        # TODO: Is this correct & should it be modifyable?
        return self.cost_to_sample

    def predict(self, option):        
        decline_value = self.compute_decline_value(option)
        take_value = self.compute_take_value(option)

        if take_value > decline_value:
            stop = True
        else:
            stop = False

        return stop
