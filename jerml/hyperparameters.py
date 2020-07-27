''' 
Generates hyperparameters for various ML models
'''

import pandas as pd


def bagging_hyperparameters(n_hyperparams, random_seed=42):
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

    Input:
    ------
    n_hyperparams: int
        Number of hyperparameters to generate

    random_seed: int, optional 
        random state used to generate hyperparameters.

    Output:
    -------
    hyperparameters: Dataframe, shape (n_hyperparams, 3)
        dataframe containing values of the parameters for an BaggingClassifier/
        BaggingRegressor class:
            - n_estimators
            - max_features
            - max_samples
    Example:
    --------
    ```python
    >>> hyperparameters = bagging_hyperparameters(n_hyperparams=3, random_seed=42)
    >>> hyperparameters.columns
    ['n_estimators', 'max_features', 'max_samples']
    ```
    '''
    pass


def decision_tree_hyperparameters(n_hyperparams, random_seed=42):
    '''
    Generates a randomized cloud of hyperparameters for an
    sklearn decision tree classifier/regressor

    Hyperparameters:
        - max_depth
            - determines max_depth of tree
            - increasing max_depth decreases bias, and increases variance
        - min_sample_split
            - minimum samples in a region required to make a split
            - increasing min_sample_split decreases bias, and increases variance
        - max_terminal_nodes
            - maximum number of leaves of the tree
            - increasing max_terminal_nodes decreases bias, and increases variance
        - min_samples_leaf
            - Ensures no leaf contains less that min_samples_leaf samples
            - increasing min_samples_leaf decreases bias, and increases variance
        - criterion
            - determines how tree decides to create a new branch
            - unclear how this affects bias-variance decomposition
        - min_impurity_decrease
            - ensures branch will only be created if it decreases the impurity
            by min_impurity_decrease

    Input:
    ------
    n_hyperparams: int
        Number of hyperparameters to generate

    random_seed: int, default=42
        random state used to generate hyperparameters.

    Output:
    -------
    hyperparameters: Dataframe, shape (n_hyperparams, 6) 
        dataframe containing values of the parameters for a 
        BaggingRegressor class:
        - max_depth
        - min_sample_split
        - max_terminal_nodes
        - min_samples_leaf
        - criterion
        - min_impurity_decrease

    Example:
    -------
    ```python
    >>> hyperparameters = decision_tree_hyperparameters(n_hyperparams=3, random_seed=42)
    >>> hyperparameters.columns
    ['max_depth', 'min_sample_split', 'max_terminal_nodes', 'min_samples_leaf', 
    'criterion', min_impurity_decrease']
    ```
    '''
    pass
