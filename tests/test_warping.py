"""
This module provides unit tests for the functions in ofco.warping.
"""

import numpy as np
from scipy.signal import correlate2d

from ofco.warping import *


def test_bilinear_interpolate():
    """
    Tests for the bilinear interpolation function.
    """
    img = np.zeros((50, 50))
    img[10:15, 20:30] = 1

    x = np.ones((50, 50))
    y = np.zeros((50, 50))
    res = np.zeros((50, 50))
    res[10:15, 19:29] = 1
    assert np.allclose(bilinear_interpolate(img, x, y), res)

    x = np.zeros((50, 50))
    y = np.ones((50, 50)) * 3
    res = np.zeros((50, 50))
    res[7:12, 20:30] = 1
    assert np.allclose(bilinear_interpolate(img, x, y), res)

    x = np.ones((50, 50)) * 2
    y = np.ones((50, 50)) * -3
    res = np.zeros((50, 50))
    res[13:18, 18:28] = 1
    assert np.allclose(bilinear_interpolate(img, x, y), res)


def test_interp2_bicubic():
    """
    Tests for the 2d bicubic interpolation function.
    """
    img = np.zeros((50, 50))
    img[10:15, 20:30] = 1
    xx, yy = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))

    xi = xx + np.zeros(img.shape)
    yi = yy + np.ones(img.shape) * 3
    res0 = np.zeros(img.shape)
    res0[7:12, 20:30] = 1
    res0[np.floor(yi) + 1 > int(img.shape[0]) - 1] = np.nan
    res0[np.floor(yi) < 0] = np.nan
    res0[np.floor(xi) + 1 > int(img.shape[1]) - 1] = np.nan
    res0[np.floor(xi) < 0] = np.nan
    output0, output1, output2 = interp2_bicubic(img, xi, yi)
    assert np.allclose(output0, res0, equal_nan=True)
    res0[np.isnan(res0)] = 0
    res1 = correlate2d(res0, np.array([[1, -8, 0, 8, -1]]) / 12, "same")
    assert np.allclose(output1, res1)
    res2 = correlate2d(res0, np.array([[1, -8, 0, 8, -1]]).transpose() / 12, "same")
    assert np.allclose(output2, res2)

    xi = xx + np.ones(img.shape) * 2
    yi = yy + np.ones(img.shape) * -3
    res0 = np.zeros(img.shape)
    res0[13:18, 18:28] = 1
    res0[np.floor(yi) + 1 > int(img.shape[0]) - 1] = np.nan
    res0[np.floor(yi) < 0] = np.nan
    res0[np.floor(xi) + 1 > int(img.shape[1]) - 1] = np.nan
    res0[np.floor(xi) < 0] = np.nan
    output0, output1, output2 = interp2_bicubic(img, xi, yi, np.array([[-0.5, 0, 0.5]]))
    assert np.allclose(output0, res0, equal_nan=True)
    res0[np.isnan(res0)] = 0
    res1 = correlate2d(res0, np.array([[-0.5, 0, 0.5]]), "same")
    assert np.allclose(output1, res1)
    res2 = correlate2d(res0, np.array([[-0.5, 0, 0.5]]).transpose(), "same")
    assert np.allclose(output2, res2)
