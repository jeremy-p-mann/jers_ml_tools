# Gets betti numbers from images

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from homcv import betti_numbers


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
        betti_numbers_list = [betti_numbers(X[i, :, :])[None,:]
                              for i in range(X.shape[0])]
        X_transformed = np.concatenate(betti_numbers_list, axis=0)
        return X_transformed

