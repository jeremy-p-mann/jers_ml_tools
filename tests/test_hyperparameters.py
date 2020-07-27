import pytest

from .context import jerml
from jerml.hyperparameters import bagging_hyperparameters
from jerml.hyperparameters import decision_tree_hyperparameters


@pytest.fixture(scope='session')
def n_hyperparams():
    return 5

@pytest.fixture(scope='session')
def bagging_params(n_hyperparams):
    return bagging_hyperparameters(n_hyperparams)
    
@pytest.fixture(scope='session')
def decision_tree_params(n_hyperparams):
    return decision_tree_hyperparameters(n_hyperparams)

def test_bagging_shape(bagging_params, n_hyperparams):
    assert bagging_params.shape[0] == n_hyperparams
    assert bagging_params.shape[1] == 3

def test_decision_tree_shape(decision_tree_params, n_hyperparams):
    assert decision_tree_params.shape[0] == n_hyperparams
    assert decision_tree_params.shape[1] == 6

# TODO: write tests to ensure correct range of values
