import pytest
from optimal_petting.utils import (
    get_relative_rank,
    random_attractiveness,
    get_sample_from_norm,
    get_optimal_thresholds
)
from scipy.stats.distributions import norm 

@pytest.mark.parametrize(
    argnames=('x', 'expected'), 
    argvalues=(
    ([0.5,0.6,], [1,1]),
    ([0.2,0.4,0.6], [1,1,1]),
    ([0.5, 0.5, 0.2], [1,1,3]),
    ([0.5, 0.5, 0.2, 0.3, 0.6], [1,1,3,3,1])
    ),
    ids=[
        'small_increasing_1',
        'small_increasing_2', 
        'simple_tie',
        'tie_with_fluctations',
    ],
)
def test_get_relative_rank(x, expected):
    assert list(get_relative_rank(x)) == expected


def test_random_attractiveness():
    attractiveness_ratings = [random_attractiveness(x) for x in [1,2,3,4,5000]]
    assert all(rating >= 0 for rating in attractiveness_ratings)
    assert all(rating <= 1 for rating in attractiveness_ratings)


def test_get_random_norm():
    print(get_sample_from_norm(10))
    pass


@pytest.mark.parametrize(
    argnames=('sequence', 'prior'),
    argvalues=(
        ([0.9,0.9,0.1,0.9, 0.9], (0.5, 0.3)),
        ([0.8,0.9, 0.75, 0.82, 0.94], (1, 0.3)),
    )
)
def test_get_optimal_thresholds(sequence, prior):

    thresholds, expected_values = get_optimal_thresholds(
        n_options=len(sequence),
        dist=norm(prior[0], prior[1]),
    )

    # strictly increasing thresholds 
    assert all(i < j for i, j in zip(thresholds, thresholds[1:]))

    # all positive (although negative values could be valid!)
    assert all(i > 0 for i in thresholds)

    # the last option is always chosen
    assert thresholds[-1] > sequence[-1]

    # print(f'seq: {sequence}, \n')
    # print(f'thresholds: {thresholds}, \n')
    # print(f'expected values: {expected_values} \n')
    # print(sequence < thresholds)
