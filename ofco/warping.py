import numpy as np
from scipy.signal import convolve2d, correlate2d


def bilinear_interpolate(img, x, y):
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


def interp2_bicubic(Z, XI, YI, Dxfilter):
    # Implementation according to Numerical Recipes
    Dyfilter = Dxfilter.transpose()
    Dxyfilter = convolve2d(Dxfilter, Dyfilter, "full")

    input_size = XI.shape

    # Reshape input coordinates into a vector
    XI = XI.flatten()
    YI = YI.flatten()

    # Bound coordinates to valid region
    sx = int(Z.shape[1])
    sy = int(Z.shape[0])

    # Neighbor coordinates
    fXI = np.floor(XI)
    cXI = fXI + 1
    fYI = np.floor(YI)
    cYI = fYI + 1

    indx = np.logical_or((fXI < 0), (cXI > sx - 1))
    indx = np.logical_or((fYI < 0), indx)
    indx = np.logical_or((cYI > sy - 1), indx)

    fXI = np.clip(fXI, 0, sx - 1).astype(np.int)
    cXI = np.clip(cXI, 0, sx - 1).astype(np.int)
    fYI = np.clip(fYI, 0, sy - 1).astype(np.int)
    cYI = np.clip(cYI, 0, sy - 1).astype(np.int)

    # Image at 4 neighbors
    Z00 = Z[(fYI, fXI)]
    Z01 = Z[(cYI, fXI)]
    Z10 = Z[(fYI, cXI)]
    Z11 = Z[(cYI, cXI)]

    # x-derivative at 4 neighbors
    DX = correlate2d(Z, Dxfilter, boundary="symm", mode="same")
    DX00 = DX[(fYI, fXI)]
    DX01 = DX[(cYI, fXI)]
    DX10 = DX[(fYI, cXI)]
    DX11 = DX[(cYI, cXI)]

    # y-derivative at 4 neighbors
    DY = correlate2d(Z, Dyfilter, boundary="symm", mode="same")
    DY00 = DY[(fYI, fXI)]
    DY01 = DY[(cYI, fXI)]
    DY10 = DY[(fYI, cXI)]
    DY11 = DY[(cYI, cXI)]

    # xy-derivative at 4 neighbors
    DXY = correlate2d(Z, Dxyfilter, boundary="symm", mode="same")
    DXY00 = DXY[(fYI, fXI)]
    DXY01 = DXY[(cYI, fXI)]
    DXY10 = DXY[(fYI, cXI)]
    DXY11 = DXY[(cYI, cXI)]

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
            Z00,
            Z10,
            Z11,
            Z01,
            DX00,
            DX10,
            DX11,
            DX01,
            DY00,
            DY10,
            DY11,
            DY01,
            DXY00,
            DXY10,
            DXY11,
            DXY01,
        ]
    )

    C = W @ V

    alpha_x = np.reshape(XI - fXI, input_size)
    alpha_y = np.reshape(YI - fYI, input_size)

    # Clip out-of-boundary pixels to boundary
    alpha_x[np.reshape(indx, input_size)] = 0
    alpha_y[np.reshape(indx, input_size)] = 0

    fXI = np.reshape(fXI, input_size)
    fYI = np.reshape(fYI, input_size)

    # Interpolation
    ZI = np.zeros(input_size)
    ZXI = np.zeros(input_size)
    ZYI = np.zeros(input_size)

    idx = 0
    alpha_x_powers = np.ones((4,) + alpha_x.shape)
    alpha_y_powers = np.ones((4,) + alpha_y.shape)
    for i in range(1, 4):
        alpha_x_powers[i] = alpha_x_powers[i - 1] * alpha_x
        alpha_y_powers[i] = alpha_y_powers[i - 1] * alpha_y

    for i in range(4):
        for j in range(4):
            ZI = (
                ZI
                + np.reshape(C[idx, :], input_size)
                * alpha_x_powers[i]
                * alpha_y_powers[j]
            )
            if i > 0:
                ZXI = (
                    ZXI
                    + i
                    * np.reshape(C[idx, :], input_size)
                    * alpha_x_powers[i - 1]
                    * alpha_y_powers[j]
                )
            if j > 0:
                ZYI = (
                    ZYI
                    + j
                    * np.reshape(C[idx, :], input_size)
                    * alpha_x_powers[i]
                    * alpha_y_powers[j - 1]
                )
            idx = idx + 1

    ZI[np.reshape(indx, input_size)] = np.nan

    return ZI, ZXI, ZYI
