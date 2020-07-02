from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import kstat


class CumulantsExtractor(BaseEstimator, TransformerMixin):
  ''' Computes cumulants less than or equal to 4'''

  def __init__(self, highest_cumulant):
    self.highest_cumulant = highest_cumulant

  def fit(self, X, y=None):
    return self

  def get_cumulants(self, v):
    kstats = np.array([kstat(data=v, n=k)
                       for k in range(1, self.highest_cumulant + 1)])
    return kstats

  def transform(self, X):
    cumulants = np.apply_along_axis(func1d=self.get_cumulants,
                                    axis=1,
                                    arr=X,
                                    )
    return cumulants
