import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import kstat


class CumulantsExtractor(BaseEstimator, TransformerMixin):
    '''
    Sci-kit learn transformer computing cumulants less than or equal to 4,
    or a specified highest cumulant.

    If the input is an image, these cumulants may be conceptualized as
    "textural" features

    Attributes:
    -----------
    highest_cumulant: int
        highest cumultant to be computed,  cannot be strictly greater than 4

    Parameters:
    -----------
    highest_cumulant_: int
        highest cumultant to be computed by the transform method.
    '''

    def __init__(self, highest_cumulant=4):
        assert highest_cumulant <= 4, 'cannot compute cumulant higher than 4'
        self.highest_cumulant_ = highest_cumulant

    def fit(self, X, y=None):
        return self

    def _get_cumulants(self, v):
        kstats = np.array([kstat(data=v, n=k)
                           for k in range(1, self.highest_cumulant_ + 1)])
        return kstats

    def transform(self, X, y=None):
        '''
        Computes cumulants of features less than the specified highest cumulant

        Input:
        ------
        X : ndarray, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        cumulants: ndarray, shape = (n_samples, highest_cumulant)
            cumulants of the empirical distribution determine by data
            along axis=1
            
        Example:
        --------
        ```python
        >>> X  = np.ones(shape = (1, 100))
        >>> cumulants_extractor = CumulantsExtractor()
        >>> cumulants_extractor.transform(X)
        [1, 0, 0, 0]
        ```
        '''
        cumulants = np.apply_along_axis(func1d=self._get_cumulants,
                                        axis=1,
                                        arr=X,
                                        )
        return cumulants
