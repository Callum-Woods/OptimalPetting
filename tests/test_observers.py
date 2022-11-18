import pytest 
from optimal_petting.observers import IdealObserver
from scipy.stats.distributions import norm 

@pytest.mark.parametrize(
    argnames='prior_obvs',
    argvalues=(
        [0.11,0.15,0.12,0.2, 0.12],
        [0.8,0.9, 0.75, 0.82, 0.94]
    ),
    ids=['scarce_prior', 'plentyful_prior']
)
def test_ideal_observer(prior_obvs):

    observer = IdealObserver(
        expected_n_applicants=5,
        cost_to_sample=0.1,
        past_observations=prior_obvs,
        reward_threshold=0.6
    )

    good_option_first = [0.76, 0.2, 0.2, 0.2, 0.2]
    good_option_last = [0.001, 0.001, 0.001, 0.76]
    good_option_middle = [0.2, 0.2, 0.75, 0.2, 0.2]

    sequences = [good_option_first, good_option_last, good_option_middle]

    for seq in sequences:
        responses = []
        for option in seq:
            responses.append(observer.predict(option))
        print(responses)
