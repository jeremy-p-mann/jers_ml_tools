''' 
Generates hyperparameters for various ML models
'''

import pandas as pd
import numpy as np


def bagging_hyperparameters(
        n_hyperparams,
        random_seed=42,
        n_estimators_support=(10**0, 10**4),
        max_features_support=(.1, .9),
        max_samples_support=(.9, 1)
        ):
    '''
    Generates a randomized cloud of hyperparameters for an
    sklearn bagging classifier/regressor

    There are three bagging hyperparameters: 
        - n_estimators: number of base estimators to be averaged
            - number of base estimators to use
            - increasing n_estimators decreases variance, mildly increases bias
        - max_features: maximum number of features to give base esimator
            - increases max_features increases variance, increases bias
                - increases effective dimension of base estimators
                - increases correlation between base estimators
        - max_samples: max number of samples used to train each base estimator
            - primarily used to decrease computational burden of training
            - decreasing max_samples increases both variance and bias

    Example
    -------
    >>> hyperparameters = bagging_hyperparameters(n_hyperparams=3, random_seed=42)
    >>> hyperparameters.columns
    ['n_estimators', 'max_features', 'max_samples']

    Parameters
    ----------
    n_hyperparams : int
        Number of hyperparameters to generate

    n_estimators_support : tuple of int, (min, max), default=(1, 10**4)
        min and max value of n_estimators hyperparatmeters.

    max_features_support : tuple of float, (min, max), default=(.1, .9)
        min and max value of max_features hyperparatmeters. More specifically,
        what percent of the features are shown to the weak learners. 

    max_samples_support : tuple, (min_value, max_value), default=(.9, 1)
        min and max value of max_samples hyperparameters. More specifically,
        what percent of the samples are shown to the weak learners. 

    random_seed_support : int, optional 
        random state used to generate hyperparameters.

    Returns
    -------
    hyperparameters : Dataframe, shape (n_hyperparams, 3)
        dataframe containing values of the parameters for an Bagging
        Classifier/Regressor class:
            - n_estimators
            - max_features
            - max_samples

    '''
    n_estimatorses = np.random.randint(
        low=n_estimators_support[0],
        high=n_estimators_support[1],
        size=n_hyperparams
        )
    max_featureses = np.random.uniform(
        low=max_features_support[0],
        high=max_features_support[1],
        size=n_hyperparams
        )
    max_sampleses = np.random.uniform(
        low=max_samples_support[0],
        high=max_samples_support[1],
        size=n_hyperparams
        )
    hyperparameter_data = {
        'n_estimators': n_estimatorses,
        'max_features': max_featureses,
        'max_samples': max_sampleses
        }
    return pd.DataFrame(data=hyperparameter_data)



def decision_tree_hyperparameters(
        n_hyperparams,
        max_depth_support=(3, 10),
        min_sample_split_support=(0, .1),
        min_samples_leaf_support=(0, .1),
        min_impurity_decrease_support=(0, .1),
        criteria=['gini', 'entropy'],
        random_seed=42,
        ):
    '''
    Generates a randomized cloud of hyperparameters for an
    sklearn decision tree classifier/regressor

    Hyperparameters:
        - max_depth
            - determines max_depth of tree
            - increasing max_depth decreases bias, and increases variance
        - min_sample_split
            - minimum samples in a region required to make a split
            - increasing min_sample_split decreases bias,
            and increases variance
        - min_samples_leaf
            - Ensures no leaf contains less that min_samples_leaf samples
            - increasing min_samples_leaf decreases bias,
            and increases variance
        - criterion
            - determines how tree decides to create a new branch
            - unclear how this affects bias-variance decomposition
        - min_impurity_decrease
            - ensures branch will only be created if it decreases the 
            impurity by min_impurity_decrease

    Example
    -------
    >>> hyperparameters = decision_tree_hyperparameters(n_hyperparams=3, random_seed=42)
    >>> hyperparameters.columns
    ['max_depth', 'min_sample_split', 'min_samples_leaf', 
    'criterion', 'min_impurity_decrease']
    
    Parameters
    ----------
    n_hyperparams : int
        Number of hyperparameters to generate.

    max_depth_support : tuple of int, (min, max), default=(3, 10)
        min and max value of the maximum tree depth hyperparatmeters.

    min_sample_split_support : tuple of float, (min, max), default=
        min and max percent of the samples must occur on a leaf 
        to make an additional split

    min_samples_leaf_support : tuple, (min, max), default=
        min and max percent of the minimum number of samples required to be
        in a leaf.

    criterion : list of string, default=
        Specifies whether to use gini and/or entropy criteria for splitting.
        
    min_impurity_decrease_support : tuple, (min, max), default=
        min and max value of mininumum decrease in the impurity caused by 
        a split. 

    random_seed: int, default=42
        random state used to generate hyperparameters.

    Returns
    -------
    hyperparameters: Dataframe, shape (n_hyperparams, 6) 
        dataframe containing values of the parameters for a 
        Bagging regressor/classifier:
        - max_depth
        - min_samples_split
        - min_samples_leaf
        - min_impurity_decrease
        - criteria

    '''
    max_depths = np.random.randint(
        low=max_depth_support[0],
        high=max_depth_support[1],
        size=n_hyperparams
        )
    min_samples_splits = np.random.uniform(
        low=min_sample_split_support[0],
        high=min_sample_split_support[1],
        size=n_hyperparams
        )
    min_samples_leaves = np.random.uniform(
        low=min_samples_leaf_support[0],
        high=min_samples_leaf_support[1],
        size=n_hyperparams
        )
    min_impurity_decreases = np.random.uniform(
        low=min_impurity_decrease_support[0],
        high=min_impurity_decrease_support[1],
        size=n_hyperparams
        )
    criteria_indices = np.random.randint(
        low=0,
        high=len(criteria),
        size=n_hyperparams
        )
    criterias = np.array(criteria)[criteria_indices]
    hyperparameter_data = {
        'max_depth': max_depths,
        'min_samples_split': min_samples_splits,
        'min_samples_leaf': min_samples_leaves,
        'min_impurity_decrease': min_impurity_decreases,
        'criteria':  criterias
        }

    return pd.DataFrame(data=hyperparameter_data)


