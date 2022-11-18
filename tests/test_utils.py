import pytest
from optimal_petting.utils import (
    get_relative_rank,
    random_attractiveness,
    get_sample_from_norm
)

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