"""
This module provides unit tests for the function in ofco.utils.
"""

import numpy as np
from scipy.signal import correlate2d

from ofco.utils import *


def test_partial_deriv():
    """
    Tests for the partial derivative function.
    """
    imgs = np.zeros((50, 50, 2))
    w = np.zeros(imgs.shape)
    output0, output1, output2 = partial_deriv(imgs, w)
    assert np.allclose(output0, np.zeros(imgs.shape[:2]))
    assert np.allclose(output1, np.zeros(imgs.shape[:2]))
    assert np.allclose(output2, np.zeros(imgs.shape[:2]))

    imgs[20:35, 35:45, 0] = 1
    imgs[20:30, 25:35, 1] = 1
    deriv_filter = np.array([[-0.5, 0, 0.5]])
    output0, output1, output2 = partial_deriv(imgs, w, deriv_filter)
    assert np.allclose(output0, imgs[:, :, 1] - imgs[:, :, 0])
    assert np.allclose(
        output1,
        (
            correlate2d(imgs[:, :, 0], deriv_filter, "same")
            + correlate2d(imgs[:, :, 1], deriv_filter, "same")
        )
        / 2,
    )
    assert np.allclose(
        output2,
        (
            correlate2d(imgs[:, :, 0], deriv_filter.transpose(), "same")
            + correlate2d(imgs[:, :, 1], deriv_filter.transpose(), "same")
        )
        / 2,
    )


def test_weighted_median():
    """
    Tests for the weighted median function.
    """
    # w = np.ones((50, 50))
    # u = np.zeros(w.shape)
    # u[10:20, 40:45] = 1
    # output = weighted_median(w, u)
    pass


def test_denoise_color_weighted_medfilt2d():
    """
    Tests for the denoise color weighted median filter 2d function.
    """
    pass


def test_detect_occlusion():
    """
    Tests for the detect occlusion function.
    """
    pass


def test_post_process():
    """
    Tests for the post processing function.
    """
    pass


def test_default_parameters():
    """
    Tests for the default parameter function.
    """
    pass


def test_midway():
    """
    Tests for the midway function.
    """
    pass


def test_crop_fit_size_center():
    """
    Tests for the crop fit size center function.
    """
    pass


def test_eigsDtD():
    """
    Tests for the eigsDtD function.
    """
    pass
