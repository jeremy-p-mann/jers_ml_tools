# Gets betti numbers from images

from homcv.betti_numbers import betti_numbers
from sklearn.base import BaseEstimator, TransformerMixin


class Bettier(BaseEstimator, TransformerMixin):
  '''
  Computes the Betti Numbers of a batch of images

  '''

  def __init__(self, threshold=.5):
    self.threshold = threshold

  def fit(self, X, y=None):
    return self

  def _get_betti_numbers(self, X):
    pass

  def transform(self, X, y=None):
    betti_numbers = None
    return None
