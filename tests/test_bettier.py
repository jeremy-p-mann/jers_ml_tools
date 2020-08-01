import pytest
import numpy as np

from .context import jerml
from jerml.transformers import Bettier

@pytest.fixture(scope='session')
def blanks():
    blanks = np.zeros(shape=(3,5,5))
    return blanks

@pytest.fixture(scope='session')
def bettier(blanks):
    bettier = Bettier(threshold_=.75).fit(blanks)
    return Bettier()

@pytest.fixture(scope='session')
def disks(blanks):
    disks = blanks.copy()
    disks[:, 2: 5, 2:5] = 1
    return disks

@pytest.fixture(scope='session')
def circles(disks):
    circles = disks.copy()
    circles[:, 3, 3] = 0
    return circles

@pytest.fixture(scope='session')
def circle_betti_numbers(bettier, blanks):
    return bettier.transform(blanks) 

def test_transform_dimensions(circle_betti_numbers):
    assert circle_betti_numbers.ndim == 2

def test_transform_n_samples(circle_betti_numbers):
    assert circle_betti_numbers.shape[0] == 3

def test_transform_n_features(circle_betti_numbers):
    assert circle_betti_numbers.shape[1] == 2

def test_blanks(bettier, blanks):
    blank_betti_numbers = np.zeros(shape=(3, 2))
    blanks_transformed = bettier.transform(blanks)
    np.testing.assert_equal(blanks_transformed, blank_betti_numbers)

def test_disks(bettier, disks):
    disk_betti_numbers = np.zeros(shape=(3, 2))
    disk_betti_numbers[:, 0] = 1
    disk_transformed = bettier.transform(disks)
    np.testing.assert_equal(disk_transformed, disk_betti_numbers)

def test_circles(bettier, circles):
    circle_betti_numbers = np.ones(shape=( 3, 2))
    circles_transformed = bettier.transform(circles)
    np.testing.assert_equal(circles_transformed, circle_betti_numbers)
