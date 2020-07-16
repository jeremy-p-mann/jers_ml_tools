'''
Transformer reshaping an ndarray
'''

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Reshaper(BaseEstimator, TransformerMixin):
    '''
    Reshapes a 2d array(e.g. from a dataframe) into a ndarray of a
    specified shape.
    '''
    
    def __init__(self, output_shape):
        self.output_shape = output_shape

    def fit(self, X, y=None):
        assert X.shape[1] == np.prod(np.array(self.output_shape)), ('output '
        'size does not match input size')
        return self

    def transform(self, X, y=None):
        ''' 
        Reshapes the array

        Input
        -----
        X: ndarray shape (n_samples, input_dim)
            input data to be transformed
        Returns
        -------
        X_trans: ndarray, shape (n_samples,) + self.output_shape
        '''
        X_transformed_shape = (X.shape[0], ) + self.output_shape
        return X.reshape(X_transformed_shape)


