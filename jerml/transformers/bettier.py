# Gets betti numbers from images

from homcv import betti_numbers
from sklearn.base import BaseEstimator, TransformerMixin


class Bettier(BaseEstimator, TransformerMixin):
    '''
    Computes the Betti Numbers of the dark regions of a batch of images

    Attributes:
    -----------
    threshold: float, optional
        The transform method computes the Betti numbers of the region 
        formed by any pixel darker than `threshold`.
    '''
    def __init__(self, threshold=.5):
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        '''
        Returns the betti numbers of the dark region of the images.
        
        Inputs
        ------
        X: ndarray, shape (n_samples, n_x, n_y)
            Batch of grayscale images.
        
        Returns
        -------
        X_transformed: ndarry, shape (n_samples, 2)
            Zeroeth and first Betti numbers of each image in the batch
        '''
        X_transformed = None
        return X_transformed

