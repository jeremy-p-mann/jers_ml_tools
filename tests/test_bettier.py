import pytest
import numpy as np

from .context import Bettier


@pytest.fixture(scope='session')
def blanks():
    blanks = np.zeros(shape=(2,10,10))
    return blanks

@pytest.fixture(scope='session')
def bettier(blanks):
    bettier = Bettier(threshold=.75).fit(blanks)
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

def test_blanks(bettier, blanks):
    blank_betti_numbers = np.zeros(shape=(2, 2))
    assert np.testing.assert_array_equal(
        bettier.transform(blanks), 
        blank_betti_numbers
        )

def test_disks(bettier, disks):
    disk_betti_numbers = np.zeros(shape=(2, 2))
    disk_betti_numbers[:, 0] = 1
    assert np.testing.assert_array_equal(
        bettier.transform(disks),
        disk_betti_numbers
        )

def test_circles(bettier, circles):
    circle_betti_numbers = np.ones(shape=(2, 2))
    assert np.testing.assert_array_equal(
        bettier.transform(circles),
        circle_betti_numbers
        )
