import logging

import numpy as np
from scipy.signal import correlate2d, medfilt2d
from scipy.ndimage.morphology import binary_dilation
import tensorflow as tf
import cv2

from .warping import interp2_bicubic

_LOGGER = logging.getLogger('ofco')


def partial_deriv(imgs, w, deriv_filter=np.array([[1, -8, 0, 8, -1]]) / 12):
    """
    Computes the partial derivatives for the three dimensions time, x and y.

    Parameters
    ----------
    imgs : numpy array
        Input images. A three dimensional array. The third dimension must
        encode time and should have length 2.
    w : numpy array
        Displacement vector field. The third dimension encodes x and y.
    deriv_filter : numpy array
        The filter used to compute the derivative.
        The default is np.array([[1, -8, 0, 8, -1]])/12.

    Returns
    -------
    It : numpy array
        Derivative with respect to time.
    Ix : numpy array
        Derivative with respect to x.
    Iy : numpy array
        Derivative with respect to y.
    """
    xx, yy = np.meshgrid(np.arange(imgs.shape[1]), np.arange(imgs.shape[0]))
    xi = xx + w[:, :, 0]
    yi = yy + w[:, :, 1]
    [warped_img, Ix, Iy] = interp2_bicubic(imgs[:, :, 1], xi, yi, deriv_filter)
    indx = np.isnan(warped_img)
    It = warped_img - imgs[:, :, 0]
    It[indx] = 0
    I1x = correlate2d(imgs[:, :, 0], deriv_filter, boundary="symm", mode="same")
    I1y = correlate2d(
        imgs[:, :, 0], deriv_filter.transpose(), boundary="symm", mode="same"
    )
    b = 0.5
    Ix = b * Ix + (1 - b) * I1x
    Iy = b * Iy + (1 - b) * I1y
    Ix[indx] = 0
    Iy[indx] = 0

    return It, Ix, Iy


def weighted_median(w, u):
    """
    Computes the weighted median uo = \min_u \sum w(i)|uo - u(i)|
    using the formula (3.13) in Y. Li and Osher
    "A New Median Formula with Applications to PDE Based Denoising"
    applied to every corresponding columns of w and u

    Parameters
    ----------
    w : numpy array
        Weights. Two dimensional.
    u : numpy array
        Data. Two dimensional.

    Returns
    -------
    uo : numpy array
        Filtered data.
    """
    H, W = u.shape
    ir = np.argsort(u, axis=0)
    # sorting again is faster than using indices
    sort_u = np.sort(u, axis=0)
    argsort_w = np.take_along_axis(w, ir, axis=0)
    k = np.ones((1, W))
    pp = -1 * np.sum(w, axis=0)
    for i in range(H - 1, -1, -1):
        pc = pp + 2 * w[i, :]
        indx = np.logical_and(pc >= 0, pp <= 0)
        k[:, indx] = H - i
        pp = pc
    k = H - k
    uo = sort_u[(k.astype(np.int), np.arange(W))]
    return uo


def denoise_color_weighted_medfilt2d(
    uv, im, occ=None, bfhsz=None, mfsz=None, sigma_i=5, sigma_x=20, fullVersion=False
):
    """
    edge region: weighted median filtering, the weights are determined by
    spatial distance, intensity distance, occlusion state
    smooth region: 
    """
    dilate_sz = [5, 5]  # dilation window size for flow edge region [5 5]

    sz = im.shape
    sz = sz[:2]

    if occ is None:
        occ = np.ones(sz)

    if bfhsz is None:
        bfhsz = 10  # half window size

    if mfsz is None:
        uvo = uv
    else:
        mfsz = int(mfsz)
        uvo = np.zeros_like(uv)
        padded_uv = np.pad(uv, ((mfsz, mfsz), (mfsz, mfsz), (0, 0)), "symmetric")
        uvo[:, :, 0] = medfilt2d(padded_uv[:, :, 0], mfsz)[mfsz:-mfsz, mfsz:-mfsz]
        uvo[:, :, 1] = medfilt2d(padded_uv[:, :, 1], mfsz)[mfsz:-mfsz, mfsz:-mfsz]

    sobelx = cv2.Sobel(uv[:, :, 0], cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(uv[:, :, 0], cv2.CV_64F, 0, 1)
    e1 = np.sqrt(sobelx ** 2 + sobely ** 2)
    e1[e1 < 0.1247] = 0
    e1[e1 >= 0.1247] = 1
    sobelx = cv2.Sobel(uv[:, :, 1], cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(uv[:, :, 1], cv2.CV_64F, 0, 1)
    e2 = np.sqrt(sobelx ** 2 + sobely ** 2)
    e2[e1 < 0.1087] = 0
    e2[e1 >= 0.1087] = 1
    e = np.logical_or(e1, e2)
    mask = binary_dilation(e, np.ones(dilate_sz))

    # below to apply WMF to all regions
    if fullVersion:
        mask = np.ones(mask.shape)
    indx_col, indx_row = np.where(mask.transpose() == 1)
    bfhsz = int(bfhsz)
    pad_u = np.pad(uv[:, :, 0], [bfhsz, bfhsz], "symmetric")
    pad_v = np.pad(uv[:, :, 1], [bfhsz, bfhsz], "symmetric")
    pad_im = np.pad(im, ((bfhsz, bfhsz), (bfhsz, bfhsz), (0, 0)), "symmetric")
    pad_occ = np.pad(occ, [bfhsz, bfhsz], "symmetric")

    H, W = pad_u.shape

    # Divide into several groups for memory reasons ~70,000 causes out of memory
    Indx_Row = indx_row
    Indx_Col = indx_col
    N = len(Indx_Row)  # number of elements to process
    n = int(4e4)  # number of elements per batch
    nB = int(np.ceil(N / n))

    for ib in range(1, nB + 1):
        istart = (ib - 1) * n  # + 1
        iend = min(ib * n, N)
        indx_row = Indx_Row[istart : iend + 1]
        indx_col = Indx_Col[istart : iend + 1]

        C, R = np.meshgrid(range(-bfhsz, bfhsz + 1), range(-bfhsz, bfhsz + 1))
        nindx = R + C * H
        cindx = indx_row + bfhsz + (indx_col + bfhsz) * H
        pad_indx = np.tile(
            nindx.transpose().flatten().reshape(-1, 1), [1, len(indx_row)]
        ) + np.tile(cindx.flatten().transpose(), [(bfhsz * 2 + 1) ** 2, 1])

        # spatial weight
        tmp = np.exp(-(C ** 2 + R ** 2) / 2 / sigma_x ** 2)
        weights = np.tile(tmp.flatten().reshape(-1, 1), [1, len(indx_row)])

        # Uncomment below: no spatial weight for test
        # weights = np.ones(weights.shape)

        # Intensity weight
        tmp_w = np.zeros(weights.shape)

        flat_cindx = cindx.flatten()
        for i in range(pad_im.shape[2]):
            tmp = pad_im[:, :, i].flatten()
            tmp_w = (
                tmp_w
                + (
                    tmp[pad_indx]
                    - np.tile(tmp[flat_cindx].transpose(), [(bfhsz * 2 + 1) ** 2, 1])
                )
                ** 2
            )

        tmp_w = tmp_w / pad_im.shape[2]

        weights = weights * np.exp(-tmp_w / 2 / sigma_i ** 2)

        # Occlusion weight
        weights = weights * pad_occ.flatten()[pad_indx]

        # Normalize
        weights = weights / np.tile(np.sum(weights, axis=0), [(bfhsz * 2 + 1) ** 2, 1])

        neighbors_u = pad_u.transpose().flatten()[pad_indx]
        neighbors_v = pad_v.transpose().flatten()[pad_indx]

        # solve weighted median filtering
        uo = uvo[:, :, 0]
        u = weighted_median(weights, neighbors_u)
        uo[(indx_row, indx_col)] = u
        vo = uvo[:, :, 1]
        v = weighted_median(weights, neighbors_v)
        vo[(indx_row, indx_col)] = v
        uvo = np.stack([uo, vo], axis=2)

    return uvo


def detect_occlusion(w, images, sigma_d=0.3, sigma_i=20):
    """
    Detect occlusion regions using flow divergence and brightness constancy
    error
    
    the output taks continous value 
    close to 0: occluded
    close to 1: nonoccluded
    
    according to Appendix A.2 Sand & Teller "Particle Video" IJCV 2008
    """
    div = np.gradient(w[:, :, 0], axis=0) + np.gradient(w[:, :, 1], axis=1)

    div[div > 0] = 0

    It, _, _ = partial_deriv(images, w)
    occ = np.exp(-div ** 2 / 2 / sigma_d ** 2) * np.exp(-It ** 2 / 2 / sigma_i ** 2)
    return occ


def post_process(w, I1, I2, sigmaS, sigmaC, occlusion_handling=False):
    stack = np.stack([I1, I2], axis=2)
    if occlusion_handling:
        occ = detect_occlusion(w, stack)
        occ = medfilt2d(occ, [9, 9])
    else:
        occ = None
    wOut = denoise_color_weighted_medfilt2d(
        w,
        stack,
        occ,
        int(np.ceil(sigmaS)),
        3,
        int(np.ceil(sigmaC)),
        int(np.ceil(sigmaS)),
        True,
    )
    return wOut


def default_parameters():
    # lmbd stands for lambda which is a build-in in python
    param = {
        "padding": 15,
        "lmbd": 800,
        "mu": 0.1,
        "nu": 0.1,
        "c2fSpacing": 1.5,
        "minSizeC2f": 10,
        "threshMatch": 70,
        "occThresh": 0.5,
        "consistencyTol": 2,
        "changeTol": 1e-3,
        "nbWarps": 1,
        "sigmaS": 1,
        "sigmaC": 200,
        "maxIters": 150,
        "iWM": 50,
    }
    return param


def refinement_parameters(img_shape):
    """
    Returns parameters that can be used to run a quick refinement of a
    given solution. The parameters it generates lead to a single pyramid
    level, i.e. the provided initial solution must be a close estimate of
    the true solution.

    Parameters
    ----------
    img_shape : tuple of integers
        Shape of the image whose warping should be refined.

    Returns
    -------
    param : dict
        Parameter dictionary for refinement.
    """
    param = default_parameters()
    param["minSizeC2f"] = max(img_shape) + 2 * param["padding"]
    return param


def midway(u_1, u_2):
    u_midway_1 = np.zeros(u_1.shape)
    u_midway_2 = np.zeros(u_2.shape)

    index_u_1 = np.argsort(u_1, axis=None)
    index_u_2 = np.argsort(u_2, axis=None)
    u_sort_1 = u_1.flatten()[index_u_1]
    u_sort_2 = u_2.flatten()[index_u_2]

    u_midway_1[np.unravel_index(index_u_1, u_1.shape)] = (u_sort_1 + u_sort_2) / 2
    u_midway_2[np.unravel_index(index_u_2, u_2.shape)] = (u_sort_1 + u_sort_2) / 2

    u_midway_1 = u_midway_1.reshape(u_1.shape)
    u_midway_2 = u_midway_2.reshape(u_2.shape)

    return [u_midway_1, u_midway_2]


def crop_fit_size_center(f, target_size):
    sf = np.array(f.shape)
    shift = np.around((sf - np.array(target_size)) / 2).astype(np.int_)
    f_out = f[
        shift[0] : shift[0] + target_size[0],
        shift[1] : shift[1] + target_size[1],
        shift[2] : shift[2] + target_size[2],
    ]
    return f_out


def eigsDtD(sx, sy, lmbd, mu):
    res = [1.0, 1.0]
    a = np.zeros((sx, sy))
    a[0, 0] = 2 / res[0] ** 2 + 2 / res[1] ** 2
    a[0, 1] = -1 / res[1] ** 2
    a[1, 0] = -1 / res[0] ** 2
    a[0, -1] = -1 / res[1] ** 2
    a[-1, 0] = -1 / res[0] ** 2
    fGtG = np.fft.fftn(a)
    return lmbd * np.real(fGtG) ** 2 + mu


def get_strategy(strategy: str = 'default'):
    """Load and return the specified Tensorflow's strategy.

    Parameters
    ----------
    strategy : string 
        either,

        * 'default' (CPU, defaults to this)
        * 'GPU' (Uses Tensorflow's MirroredStrategy)
        * 'TPU'
        * 'GPU:<gpu-index>',
            * If ``strategy`` is 'GPU', then use all GPUs.
            * If ``strategy`` is 'GPU:0', then use GPU 0.
            * If ``strategy`` is 'GPU:1', then use GPU 1.
            * etc.

    Returns
    -------
    :func:`tf.distribute.Strategy`
        Tensorflow strategy
    """
    # print device info
    _LOGGER.info(f"Num Physical GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
    _LOGGER.info(f"Num Logical  GPUs Available: {len(tf.config.list_logical_devices('GPU'))}")
    _LOGGER.info(f"Num TPUs Available: {len(tf.config.list_logical_devices('TPU'))}")

    if not tf.test.is_built_with_cuda():
        logging.warning('Tensorflow is not built with GPU support!')

    # try to allow growth in case other people are using the GPUs
    for gpu in tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except:
            _LOGGER.warning(f'GPU device "{gpu}" is already initialized.')

    # choose strategy
    if strategy.lower() == 'tpu' and tf.config.list_physical_devices('TPU'):
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
        tf.config.experimental_connect_to_cluster(resolver)
        # This is the TPU initialization code that has to be at the beginning.
        tf.tpu.experimental.initialize_tpu_system(resolver)

        strategy = tf.distribute.TPUStrategy(resolver)
        _LOGGER.info(r'using TPU strategy.')
    if strategy.lower()[:3] == 'gpu' and tf.config.list_physical_devices('GPU'):
        if len(strategy) > 3:
            strategy = tf.distribute.MirroredStrategy([strategy])
        else:
            strategy = tf.distribute.MirroredStrategy()
        _LOGGER.info(r'using GPU "MirroredStrategy" strategy.')
    else:
        # use default strategy
        strategy = tf.distribute.get_strategy()
        _LOGGER.info(r'using default strategy.')

    return strategy