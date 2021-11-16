from scipy.ndimage import gaussian_filter
import math
import logging
from skimage import transform
import numpy as np
import tensorflow as tf

from .utils import eigsDtD, partial_deriv, post_process

_LOGGER = logging.getLogger('ofco')


def optical_flow_estimation(I1, I2, sz0, param, verbose=False, initial_w=None):

    sigmaPreproc = 0.9
    I1 = gaussian_filter(I1, sigmaPreproc, mode="mirror")
    I2 = gaussian_filter(I2, sigmaPreproc, mode="mirror")

    deriv_filter = np.array([[1, -8, 0, 8, -1]]) / 12.0

    # coarse-to-fine parameters
    minSizeC2f = param["minSizeC2f"]
    c2fLevels = int(
        math.ceil(
            math.log(minSizeC2f / max(I1.shape)) / math.log(1 / param["c2fSpacing"])
        )
    )
    c2fLevels = max(c2fLevels, 1)

    factor = math.sqrt(2)
    smooth_sigma = math.sqrt(param["c2fSpacing"]) / factor
    I1C2f = reversed(
        list(
            transform.pyramid_gaussian(
                I1,
                max_layer=c2fLevels - 1,
                downscale=param["c2fSpacing"],
                sigma=smooth_sigma,
                multichannel=False,
            )
        )
    )
    I2C2f = reversed(
        list(
            transform.pyramid_gaussian(
                I2, c2fLevels - 1, param["c2fSpacing"], smooth_sigma, multichannel=False
            )
        )
    )

    lambdaC2f = np.empty(c2fLevels)
    sigmaSSegC2f = np.empty(c2fLevels)

    for i in range(c2fLevels - 1, -1, -1):
        lambdaC2f[i] = param["lmbd"]
        sigmaSSegC2f[i] = param["sigmaS"] / (param["c2fSpacing"] ** i)

    # Initationlization
    wl = np.zeros((I1.shape[0], I1.shape[1], 2))
    if initial_w is not None:
        if not np.all(initial_w.shape == wl.shape):
            raise ValueError(
                f"initial_w has wrong shape. Given shape is {initial_w.shape} expected {wl.shape}"
            )
        wl = initial_w

    # Coarse to fine
    for l, I1, I2 in zip(range(c2fLevels - 1, -1, -1), I1C2f, I2C2f):
        # Scaled data
        sigmaS = sigmaSSegC2f[l]
        lmbd = lambdaC2f[l]

        # Resacle flow
        ratio = I1.shape[0] / wl[:, :, 0].shape[0]
        ul = transform.resize(wl[:, :, 0], I1.shape, order=3) * ratio
        ratio = I1.shape[1] / wl[:, :, 1].shape[1]
        vl = transform.resize(wl[:, :, 1], I1.shape, order=3) * ratio
        wl = np.dstack((ul, vl))
        sz0 = np.floor(np.array(sz0) * ratio)

        mu = param["mu"]
        nu = param["nu"]
        munu = mu + nu

        eigs_DtD = eigsDtD(I1.shape[0], I1.shape[1], lmbd, mu)

        for iWarp in range(param["nbWarps"]):
            w_prev = wl
            dwl = np.zeros((wl.shape))

            alpha = np.zeros((I1.shape[0], I1.shape[1], 2))
            z = np.zeros((I1.shape[0], I1.shape[1], 2))

            # Pre-computations
            It, Ix, Iy = partial_deriv(np.stack([I1, I2], axis=2), wl, deriv_filter)

            Igrad = Ix ** 2 + Iy ** 2 + 1e-3
            thresh = Igrad / munu

            # Main iterations loop
            for it in range(param["maxIters"]):
                # Data update
                r1 = z - wl - alpha / mu
                t = (mu * r1 + nu * dwl) / munu

                rho = It + t[:, :, 0] * Ix + t[:, :, 1] * Iy

                idx1 = rho < -thresh
                idx2 = rho > thresh
                idx3 = np.abs(rho) <= thresh
                # idx3 = tf.cast(tf.math.abs(rho) <= thresh, tf.float32)

                dwl = t

                dwl[:, :, 0] += Ix * idx1 / munu
                dwl[:, :, 1] += Iy * idx1 / munu
                dwl[:, :, 0] -= Ix * idx2 / munu
                dwl[:, :, 1] -= Iy * idx2 / munu
                dwl[:, :, 0] -= (rho * Ix / Igrad) * idx3
                dwl[:, :, 1] -= (rho * Iy / Igrad) * idx3

                w = wl + dwl

                # Regularization update
                muwalpha = mu * w + alpha
                ## NumPy
                # z[:, :, 0] = np.real(
                #     np.fft.ifft2(np.divide(np.fft.fft2(muwalpha[:, :, 0]), eigs_DtD))
                # )
                # z[:, :, 1] = np.real(
                #     np.fft.ifft2(np.divide(np.fft.fft2(muwalpha[:, :, 1]), eigs_DtD))
                # )
                ## Tensorflow
                z[:, :, 0] = np.real(
                    tf.signal.ifft2d(tf.math.divide(tf.signal.fft2d(muwalpha[:, :, 0]), eigs_DtD))
                )
                z[:, :, 1] = np.real(
                    tf.signal.ifft2d(tf.math.divide(tf.signal.fft2d(muwalpha[:, :, 1]), eigs_DtD))
                )

                # Lagrange parameters update
                alpha = alpha + mu * (w - z)

                # Post-processing
                if (it % param["iWM"]) == 0:
                    w0 = post_process(w, I1, I2, sigmaS, param["sigmaC"])
                    dwl = w0 - wl
                    w = wl + dwl

                # End of iterations checking
                norm_w_prev = np.linalg.norm(w_prev.flatten())
                # norm_w_prev = tf.norm(w_prev.flatten())
                if norm_w_prev == 0:
                    w_prev = w
                    continue
                else:
                    change = (
                        np.linalg.norm(w.flatten() - w_prev.flatten()) / norm_w_prev
                    )
                    if change < param["changeTol"]:
                        if verbose:
                            _LOGGER.info(f'Converged scale {l} and warp {iWarp} in {it=} iterations')
                        break
                    w_prev = w

            wl = wl + dwl
            wl = post_process(wl, I1, I2, sigmaS, param["sigmaC"])
    return wl
