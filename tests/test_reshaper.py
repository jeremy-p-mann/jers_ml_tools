import numpy as np
import pytest

from .context import jerml
from jerml.transformers import Reshaper


@pytest.fixture(scope='session')
def X_shape():
    return (10, 150)

@pytest.fixture(scope='session')
def X(X_shape):
    X = np.ones(shape=X_shape)
    return X

@pytest.fixture(scope='session')
def output_shape():
    return (10, 15)

@pytest.fixture(scope='session')
def reshaper(X, output_shape):
    reshaper = Reshaper(output_shape=output_shape).fit(X)
    return reshaper
    
@pytest.fixture(scope='session')
def X_transformed(X, reshaper):
    X_transformed = reshaper.transform(X)
    return X_transformed

def test_output_n_samples(X_transformed, X):
    assert X_transformed.shape[0] == X.shape[0]

def test_output_shape(output_shape, X_transformed):
    assert X_transformed.shape[1:] == output_shape 
