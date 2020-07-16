from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin


class GrayScaler(BaseEstimator, TransformerMixin):
    '''
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

        Inputs:
        -------
        X: ndarray, shape (n_samples, x_dim, y_dim, n_color_channels)
            Array of n_samples images, of size (x_dim, y_dim) with
            n_color_channels

        Returns:
        --------
        X_grayscale: ndarray, shape (n_samples, x_dim, y_dim)
            Array of n_samples grayscale images of the same size as the
            input X.
        '''

        image_dimensions = (X.shape[1], X.shape[2])
        n_samples = X.shape[0]

        X_flat = self._flatten(X)
        X_grayscale_flat = self.pca.transform(X_flat)
        X_grayscale = self._unflatten(X_grayscale_flat, n_samples, image_dimensions)
        return X_grayscale

