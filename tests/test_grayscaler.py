import numpy as np
import pytest

from .context import GrayScaler


@pytest.fixture(scope='session')
def grayscaler():
    grayscaler_instance = GrayScaler()
    return grayscaler_instance


@pytest.fixture(scope='session')
def images():
    batch_dimensions = (2, 10, 15, 3)
    images = np.zeros(shape=batch_dimensions)
    return images


@pytest.fixture(scope='session')
def flattened_images(grayscaler, images):
    flattened_images = grayscaler._flatten(images)
    return flattened_images


@pytest.fixture(scope='session')
def unflattened_images(grayscaler):
    flat_image = np.zeros(shape=(5  , 10*15))
    return grayscaler._unflatten(flat_image, 5, (10,15))


def test_grayscaler_flatten_shape(images, flattened_images):
    total_n_pixels = images.shape[0] * images.shape[1] * images.shape[2]
    assert flattened_images.shape[0] == total_n_pixels


def test_grayscaler_flatten_color_dimension( flattened_images):
    '''flattened image flattened to 1 dimension'''
    assert flattened_images.ndim == 2


def test_grayscaler_unflatten_image_dimensions(unflattened_images):
    assert unflattened_images.shape[1] == 10
    assert unflattened_images.shape[2] == 15


def test_grayscaler_unflatten_n_samples(flattened_images, 
                                        unflattened_images):
    assert unflattened_images.shape[0] == 5


def test_grayscaler_transformer(grayscaler):
    grayscale_image_batch_shape =  (2, 10, 15, 1)

    blank_images  = np.zeros(shape=grayscale_image_batch_shape)
    
    images_of_dot = blank_images.copy()
    images_of_dot[:, 5, 5, :] = 1

    color_channel_list = [blank_images, images_of_dot, blank_images]
    color_grayscale_disks = np.concatenate(color_channel_list,
                                          axis=3)

    grayscaler.fit(color_grayscale_disks)
    grayscaled_images_of_dot = grayscaler.transform(color_grayscale_disks)
    
    # first test showed they agreed up to 2 decimal places
    np.testing.assert_almost_equal(images_of_dot[:, :, :, 0],
                                   grayscaled_images_of_dot,
                                   decimal=2)



