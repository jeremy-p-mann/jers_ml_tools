import numpy as np
from .context import jerml
from jerml.transformers import CumulantsExtractor

def test_cumulants_dirac_distribution():
    cumulants_extractor = CumulantsExtractor()
    X = np.ones(shape=(1, 10))

    cumulants_pred = cumulants_extractor.transform(X)
    cumulants_true = np.array([1, 0, 0,  0]).reshape(1, -1)

    np.testing.assert_equal(cumulants_pred, cumulants_true)


def test_cumulants_normal_distribution():
    cumulants_extractor = CumulantsExtractor()
    np.random.seed(42)
    X = np.random.normal(0, 1, (1, 10**5))

    cumulants_pred = cumulants_extractor.transform(X)
    cumulants_true = np.array([0, 1, 0,  0]).reshape(1, -1)

    np.testing.assert_allclose(
        cumulants_pred,
        cumulants_true,
        rtol=0,
        atol=1e-1
        )
