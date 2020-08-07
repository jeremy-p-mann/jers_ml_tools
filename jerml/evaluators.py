``` 
Evaluates the generalization error of a learning algorithm using 
cross validation.
```

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold


def cv_confusion_matrices(
        pipeline,
        X,
        y,
        train_test_ratio=.2,
        n_repeats=10,
        random_seed=42
        ):
    '''
    Generates train/test confusion matrices via cross validation

    Input
    -----
    pipeline : Pipeline
        ML pipeline to be evaluated. 

    X : ndarray, shape (n_samples, ... )
        Array of features

    y : ndarray, shape (n_samples, ... ) 
        Array of labels
    
    train_test_ratio : float, default=.2
        Determines what proportion of the data will be used to evaluate the 
        pipeline.

    n_repeats : int, default =5
        number of splits to be generated.        

    random_seed : int, default=42
        Controls the generation of the splits, ensuring reproducible outputs

    Returns
    -------
    training_confusion_matrices ndarray, shape=(k, n_labels, n_labels)
        confusion matrices evaluated on training data for every split

    test_confusion_matrices: ndarray, shape=(k, n_labels, n_labels)
        confusion matrices evaluated on test data for every split

    Example
    -------
    >>> from sklearn.linear_model import LogisticRegression
    >>> X = [[1, 2], [2, 1], [1, 3], [0, 0]]
    >>> y = [0, 1, 1, 0]
    >>> pipeline = LogisticRegression() 
    >>> training_cm, test_cm = cv_confusion_matrices(pipeline, X, y,
    ...     n_repeats=1, )
    [[0, 1], [0, 1]], [[0, 1,], [0, 1]]
    '''
    pass

