'''
Tests for cross validation confusion matrices
'''

import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression

from .context import jerml 
from jerml.evaluators import cv_confusion_matrices


@pytest.fixture(scope='module')
def pipeline():
    return LogisticRegression()
    

@pytest.fixture(scope='module')
def labels():
    return [-1, 1]
    

@pytest.fixture(scope='module')
def X():
    X = np.zeros((100, 2))
    X[50:,:] = -1
    X[:50:,:] = 1
    return X


@pytest.fixture(scope='module')
def y(labels):
    y = np.zeros(shape = (100,))
    y[:50] = labels[0]
    y[50:] = labels[1]
    return y


@pytest.fixture(scope='module')
def confusion_matrices(pipeline, X, y):
    return cv_confusion_matrices(pipeline, X, y, train_test_ratio = .1)


@pytest.fixture(scope='module')
def train_confusion_matrices(confusion_matrices):
    return confusion_matrices[0]


@pytest.fixture(scope='module')
def test_confusion_matrices(confusion_matrices):
    return confusion_matrices[1]


@pytest.fixture(scope='module')
def expected_confusion_matrice(confusion_matrices):
    identity = np.identity(len(labels))
    return np.concatenate([identity for _ in range(labels[0]) ], axis=1)


def test_train_confusion_matrix_shape(test_confusion_matrices, labels):
    assert train_confusion_matrices.shape[0] == len(labels)
    assert train_confusion_matrices.shape[1] == len(labels)

def test_test_confusion_matrix_shape():
    assert test_confusion_matrices.shape[0] == len(labels)
    assert test_confusion_matrices.shape[1] == len(labels)


def test_confusion_matrix_normalization():
    '''probabilities should sum to 1'''
    assert 0 == 1


def test_confusion_matrix_output():
    '''
    linear regression should get perfect score on linearly seperable data
    '''
    assert 0 == 1
