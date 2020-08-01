import pytest

from .context import jerml
from jerml.hyperparameters import bagging_hyperparameters
from jerml.hyperparameters import decision_tree_hyperparameters


@pytest.fixture(scope='session')
def n_hyperparams():
    return 100


@pytest.fixture(scope='session')
def percent_support():
    return (.1, .9)


@pytest.fixture(scope='session')
def value_support():
    return (1, 100)


@pytest.fixture(scope='session')
def bagging_params(n_hyperparams, percent_support, value_support):
    return bagging_hyperparameters(
            n_hyperparams,
            n_estimators_support=value_support,
            max_features_support=percent_support,
            max_samples_support=percent_support
            )
    

@pytest.fixture(scope='session')
def decision_tree_params(n_hyperparams, percent_support, value_support):
    return decision_tree_hyperparameters(
            n_hyperparams, 
            max_depth_support=value_support,
            min_sample_split_support=percent_support,
            min_samples_leaf_support=percent_support,
            min_impurity_decrease_support=percent_support
            )


def test_bagging_shape(bagging_params, n_hyperparams):
    assert bagging_params.shape[0] == n_hyperparams
    assert bagging_params.shape[1] == 3


def test_decision_tree_shape(decision_tree_params, n_hyperparams):
    assert decision_tree_params.shape[0] == n_hyperparams
    assert decision_tree_params.shape[1] == 5


def test_bagging_percent_supports(bagging_params, percent_support):
    columns = ['max_features', 'max_samples']
    bagging_percent_params = bagging_params[columns]
    assert bagging_percent_params.min().min() >= percent_support[0]
    assert bagging_percent_params.max().max() <= percent_support[1]


def test_bagging_value_supports(bagging_params, value_support):
    columns = ['n_estimators']
    bagging_value_params = bagging_params[columns]
    assert bagging_value_params.min().min() >= value_support[0]
    assert bagging_value_params.max().max() <= value_support[1]

 
def test_decision_tree_percent_supports(decision_tree_params, percent_support):
    columns = [
            'min_samples_split',
            'min_samples_leaf',
            'min_impurity_decrease'
            ]
    decision_tree_percent_params = decision_tree_params[columns]
    assert decision_tree_percent_params.min().min() >= percent_support[0]
    assert decision_tree_percent_params.max().max() <= percent_support[1]


def test_decision_tree_value_supports(decision_tree_params, value_support):
    columns = ['max_depth'] 
    decision_tree_value_params = decision_tree_params[columns]
    assert decision_tree_value_params.min().min() >= value_support[0]
    assert decision_tree_value_params.max().max() <= value_support[1]


def test_decision_tree_criteria(decision_tree_params):
    criteria = set(decision_tree_params['criteria'])
    assert 'gini' in criteria
    assert 'entropy' in criteria


