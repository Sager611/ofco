import numpy as np
from scipy.signal import convolve2d, correlate2d


def bilinear_interpolate(img, x, y):
    """
    This function performs a bilinear interpolation on an image for a given
    displacement vector field with components x and y.

    Parameters
    ----------
    img : numpy array
        Input image.
    x : numpy array
        X components of the displacement vector field in pixels.
        Same dimensions as the input image.
    y : numpy array
        Y components of the displacement vector field in pixels.
        Same dimensions as the input image.

    Returns
    -------
    warped_image : numpy array
        Image warped with the given vector field.
    """
    xx, yy = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
    x = np.asarray(xx + x)
    y = np.asarray(yy + y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, img.shape[1] - 1)
    x1 = np.clip(x1, 0, img.shape[1] - 1)
    y0 = np.clip(y0, 0, img.shape[0] - 1)
    y1 = np.clip(y1, 0, img.shape[0] - 1)

    Ia = img[y0, x0]
    Ib = img[y1, x0]
    Ic = img[y0, x1]
    Id = img[y1, x1]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return wa * Ia + wb * Ib + wc * Ic + wd * Id


def interp2_bicubic(img, xi, yi, dx_filter=np.array([[1, -8, 0, 8, -1]]) / 12):
    """
    This function computes the bicubic 2d interpolation of a given image
    at point xi and yi and the partial derivatives at these locations.

    Parameters
    ----------
    img : numpy array
        Input image.
    xi : numpy array
        X coordinates/indices of interpolation points.
    yi : numpy array
        Y coordinates/indices of interpolation points.
    dx_filter : numpy array
        Filter used to calculate the x derivative.
        Default is np.array([[1, -8, 0, 8, -1]]) / 12.

    Returns
    -------
    img_interp : numpy array
        Interpolated image.
    img_interp_dx : numpy array
        The partial derivative of the interpolated image with respect to x.
    img_interp_dy
        The partial derivative of the interpolated image with respect to y.
    """
    # Implementation according to Numerical Recipes
    dy_filter = dx_filter.transpose()
    dxy_filter = convolve2d(dx_filter, dy_filter, "full")

    input_size = xi.shape

    # Reshape input coordinates into a vector
    xi = xi.flatten()
    yi = yi.flatten()

    # Bound coordinates to valid region
    sx = int(img.shape[1])
    sy = int(img.shape[0])

    # Neighbor coordinates
    fxi = np.floor(xi)
    cxi = fxi + 1
    fyi = np.floor(yi)
    cyi = fyi + 1

    indx = np.logical_or((fxi < 0), (cxi > sx - 1))
    indx = np.logical_or((fyi < 0), indx)
    indx = np.logical_or((cyi > sy - 1), indx)

    fxi = np.clip(fxi, 0, sx - 1).astype(np.int)
    cxi = np.clip(cxi, 0, sx - 1).astype(np.int)
    fyi = np.clip(fyi, 0, sy - 1).astype(np.int)
    cyi = np.clip(cyi, 0, sy - 1).astype(np.int)

    # Image at 4 neighbors
    img00 = img[(fyi, fxi)]
    img01 = img[(cyi, fxi)]
    img10 = img[(fyi, cxi)]
    img11 = img[(cyi, cxi)]

    # x-derivative at 4 neighbors
    dx = correlate2d(img, dx_filter, boundary="symm", mode="same")
    dx00 = dx[(fyi, fxi)]
    dx01 = dx[(cyi, fxi)]
    dx10 = dx[(fyi, cxi)]
    dx11 = dx[(cyi, cxi)]

    # y-derivative at 4 neighbors
    dy = correlate2d(img, dy_filter, boundary="symm", mode="same")
    dy00 = dy[(fyi, fxi)]
    dy01 = dy[(cyi, fxi)]
    dy10 = dy[(fyi, cxi)]
    dy11 = dy[(cyi, cxi)]

    # xy-derivative at 4 neighbors
    dxy = correlate2d(img, dxy_filter, boundary="symm", mode="same")
    dxy00 = dxy[(fyi, fxi)]
    dxy01 = dxy[(cyi, fxi)]
    dxy10 = dxy[(fyi, cxi)]
    dxy11 = dxy[(cyi, cxi)]

    W = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [-3, 0, 0, 3, 0, 0, 0, 0, -2, 0, 0, -1, 0, 0, 0, 0],
            [2, 0, 0, -2, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, -3, 0, 0, 3, 0, 0, 0, 0, -2, 0, 0, -1],
            [0, 0, 0, 0, 2, 0, 0, -2, 0, 0, 0, 0, 1, 0, 0, 1],
            [-3, 3, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, -2, -1, 0, 0],
            [9, -9, 9, -9, 6, 3, -3, -6, 6, -6, -3, 3, 4, 2, 1, 2],
            [-6, 6, -6, 6, -4, -2, 2, 4, -3, 3, 3, -3, -2, -1, -1, -2],
            [2, -2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 1, 1, 0, 0],
            [-6, 6, -6, 6, -3, -3, 3, 3, -4, 4, 2, -2, -2, -2, -1, -1],
            [4, -4, 4, -4, 2, 2, -2, -2, 2, -2, -2, 2, 1, 1, 1, 1],
        ]
    )

    V = np.array(
        [
            img00,
            img10,
            img11,
            img01,
            dx00,
            dx10,
            dx11,
            dx01,
            dy00,
            dy10,
            dy11,
            dy01,
            dxy00,
            dxy10,
            dxy11,
            dxy01,
        ]
    )

    C = W @ V

    alpha_x = np.reshape(xi - fxi, input_size)
    alpha_y = np.reshape(yi - fyi, input_size)

    # Clip out-of-boundary pixels to boundary
    alpha_x[np.reshape(indx, input_size)] = 0
    alpha_y[np.reshape(indx, input_size)] = 0

    fxi = np.reshape(fxi, input_size)
    fyi = np.reshape(fyi, input_size)

    # Interpolation
    img_interp = np.zeros(input_size)
    img_interp_dx = np.zeros(input_size)
    img_interp_dy = np.zeros(input_size)

    idx = 0
    alpha_x_powers = np.ones((4,) + alpha_x.shape)
    alpha_y_powers = np.ones((4,) + alpha_y.shape)
    for i in range(1, 4):
        alpha_x_powers[i] = alpha_x_powers[i - 1] * alpha_x
        alpha_y_powers[i] = alpha_y_powers[i - 1] * alpha_y

    for i in range(4):
        for j in range(4):
            img_interp = (
                img_interp
                + np.reshape(C[idx, :], input_size)
                * alpha_x_powers[i]
                * alpha_y_powers[j]
            )
            if i > 0:
                img_interp_dx = (
                    img_interp_dx
                    + i
                    * np.reshape(C[idx, :], input_size)
                    * alpha_x_powers[i - 1]
                    * alpha_y_powers[j]
                )
            if j > 0:
                img_interp_dy = (
                    img_interp_dy
                    + j
                    * np.reshape(C[idx, :], input_size)
                    * alpha_x_powers[i]
                    * alpha_y_powers[j - 1]
                )
            idx = idx + 1

    img_interp[np.reshape(indx, input_size)] = np.nan

    return img_interp, img_interp_dx, img_interp_dy
