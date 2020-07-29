'''
This module contains a medley of sklearn transformers which can be integrated
into a pipeline.
'''

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from scipy.stats import kstat
from homcv import betti_numbers


class CumulantsExtractor(BaseEstimator, TransformerMixin):
    '''Scikit-Learn transformer computing cumulants of the features.

    Cumulants are universal numerical invariants of probability
    distributions. Their interpretation is context dependent. For example,
    if the input is an image, these cumulants may be conceptualized as
    "textural" features. 

    Note that this transformer can only compute the first 4 cumulants.

    Example
    -------
    >>> X  = np.ones(shape = (1, 100))
    
    This distribution is entirely "deterministic", and we should therefore 
    expect it to have no cumulants higher that 1, and have an expectation
    value of 1.

    >>> cumulants_extractor = CumulantsExtractor()
    >>> cumulants_extractor.transform(X)
    [1, 0, 0, 0]

    Attributes
    ----------
    highest_cumulant_ : int
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

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        cumulants: ndarray, shape = (n_samples, highest_cumulant)
            cumulants of the empirical distribution determine by data
            along axis=1

        '''
        cumulants = np.apply_along_axis(func1d=self._get_cumulants,
                                        axis=1,
                                        arr=X,
                                        )
        return cumulants


class GrayScaler(BaseEstimator, TransformerMixin):
    '''Transforms a color image into grayscale.

    Transforms a batch color images into a batch of grayscale images
    using 1-component PCA.
    '''

    def __init__(self):
        self.pca = PCA(n_components=1)
        pass

    def _flatten(self, X):
        '''
        Flattens the image so that it can be transformed into a form
        PCA can transform
        '''
        assert X.ndim == 4, "batch must be 4 dimensional"
        n_color_channels = X.shape[-1]

        X_flat = X.reshape(-1, n_color_channels)
        return X_flat

    def _unflatten(self, X_grayscale_flat, n_samples, image_dimensions):
        '''
        Unflattens image, making it have shape (n_samples, n_x, n_y)
        '''
        X_unflat = X_grayscale_flat.reshape(n_samples,
                                            image_dimensions[0],
                                            image_dimensions[1])
        return X_unflat

    def fit(self, X, y=None):
        '''
        Fits a 1-component PCA on the distributions of colors of all the
        pixels in the entire batch of images.
        '''

        X_flat = self._flatten(X)
        self.pca.fit(X_flat)

        return self

    def transform(self, X, y=None):
        '''
        Finds a gray-scale approximation to a batch of images
        using 1-component PCA in color space.

        Parameters
        ----------
        X: ndarray, shape (n_samples, x_dim, y_dim, n_color_channels)
            Array of n_samples images, of size (x_dim, y_dim) with
            n_color_channels

        Returns
        -------
        X_grayscaled: ndarray, shape (n_samples, x_dim, y_dim)
            Array of n_samples grayscale images of the same size as the
            input X.
        '''

        image_dimensions = (X.shape[1], X.shape[2])
        n_samples = X.shape[0]

        X_flat = self._flatten(X)
        X_grayscale_flat = self.pca.transform(X_flat)
        X_grayscaled = self._unflatten(X_grayscale_flat, n_samples, image_dimensions)
        return X_grayscaled


class Reshaper(BaseEstimator, TransformerMixin):
    ''' Reshapes a 2d array into a ndarray of a specified shape.

    Attributes
    ----------
    output_shape : tuple of int
        shape of the output array
    '''
    
    def __init__(self, output_shape):
        self.output_shape = output_shape

    def fit(self, X, y=None):
        assert X.shape[1] == np.prod(np.array(self.output_shape)), ('output '
        'size does not match input size')
        return self

    def transform(self, X, y=None):
        ''' Reshapes the array

        Parameters
        ----------
        X : ndarray, shape (n_samples, input_dim)
            input data to be transformed

        Returns
        -------
        X_reshaped: ndarray, shape (n_samples,) + self.output_shape
            Reshaped array
        '''
        X_transformed_shape = (X.shape[0], ) + self.output_shape
        return X.reshape(X_transformed_shape)


class Bettier(BaseEstimator, TransformerMixin):
    '''Computes the Betti Numbers of the dark regions of a batch of images

    Attributes
    ----------
    threshold_ : float, optional
        The transform method computes the Betti numbers of the region 
        formed by any pixel darker than `threshold`.
    '''
    def __init__(self, threshold_=.5):
        self.threshold_ = threshold_

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        '''
        Returns the betti numbers of the dark region of the images.
        
        Parameters
        ----------
        X : ndarray, shape (n_samples, n_x, n_y)
            Batch of grayscale images.
        
        Returns
        -------
        X_transformed : ndarry, shape (n_samples, 2)
            Zeroeth and first Betti numbers of each image in the batch
        '''
        betti_numbers_list = [betti_numbers(X[i, :, :], self.threshold_)[None,:]
                              for i in range(X.shape[0])]
        X_transformed = np.concatenate(betti_numbers_list, axis=0)
        return X_transformed

