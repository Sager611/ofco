import os
import multiprocessing
from timeit import default_timer as timer
import numpy as np
from skimage import io
from numba import jit, prange

from .utils import default_parameters, midway, crop_fit_size_center
from .warping import bilinear_interpolate
from .of_l1_l2_fm_admm import of_l1_l2_fm_admm


def compute_motion(I1, I2, param):
    sz0 = I1.shape

    I1 = np.pad(I1, [15, 15], "edge")
    I2 = np.pad(I2, [15, 15], "edge")

    # Optical flow
    w = of_l1_l2_fm_admm(I1, I2, sz0, param)

    w = crop_fit_size_center(w, [sz0[0], sz0[1], 2])
    return w


def parallel_compute_motion(t):
    i2 = global_stack1_rescale[t + 1, :, :]
    [i10, i2] = midway(global_i1, i2)
    return compute_motion(i10, i2, global_param)


def motion_compensate(
    stack1, stack2, output1, output2, frames, param, verbose=False, parallel=True
):
    start = timer()
    stack1 = stack1[frames, :, :].astype(np.float64)
    stack1_rescale = (
        (stack1 - np.amin(stack1)) / (np.amax(stack1) - np.amin(stack1)) * 255
    )
    stack2 = stack2[frames, :, :].astype(np.float64)
    end = timer()
    if verbose:
        print("Time it took to normalize images {}".format(end - start))

    # Motion estimation
    start = timer()
    w = np.zeros(
        [
            stack1_rescale.shape[0] - 1,
            stack1_rescale.shape[1],
            stack1_rescale.shape[2],
            2,
        ]
    )

    i1 = stack1_rescale[frames[0], :, :]
    if parallel:
        global global_stack1_rescale
        global_stack1_rescale = stack1_rescale
        global global_i1
        global_i1 = i1
        global global_param
        global_param = param
        pool = multiprocessing.Pool()
        output = pool.map(parallel_compute_motion, range(len(frames) - 1))
        w = np.array(output)
    else:
        for t in range(len(frames) - 1):
            if verbose:
                print("Frame {}\n".format(t))
            i2 = stack1_rescale[t + 1, :, :]
            [i10, i2] = midway(i1, i2)
            w[t, :, :, :] = compute_motion(i10, i2, param)
    end = timer()
    if verbose:
        print("Time it took to compute motion field w {}".format(end - start))

    start = timer()
    stack1_warped = stack1
    stack2_warped = stack2
    for t in range(len(frames) - 1):
        stack1_warped[t + 1, :, :] = bilinear_interpolate(
            stack1[t + 1, :, :], w[t, :, :, 0], w[t, :, :, 1]
        )
        stack2_warped[t + 1, :, :] = bilinear_interpolate(
            stack2[t + 1, :, :], w[t, :, :, 0], w[t, :, :, 1]
        )
    end = timer()
    if verbose:
        print("Time it took to warp images {}".format(end - start))

    start = timer()
    io.imsave(output1, stack1_warped.astype(np.float32))
    io.imsave(output2, stack2_warped.astype(np.float32))
    end = timer()
    if verbose:
        print("Time it took to save images {}".format(end - start))


def main(stack1, stack2, output_dir, frames=-1, verbose=False, **kwargs):
    """
    Parameters
    ----------
    stack1 : string
        Path to stack with constant brightness (e.g. tdTom).
    stack2 : string
        Path to stack with functional information (e.g. GCamP).
    output_dir : string
        Path to the output directory. If it does not exist it is
        created.
    frames : int or list of int, optional
        Frames that are motion corrected.
        Default is -1, which means all frames in the stack are considered.
    verbose : boolean
        Default False.

    Additional keyword arguments can be used to change the parameters of the
    algorithm. For a description of the parameters see the default_parameters.
    """
    # Parameters
    param = default_parameters()
    for key, value in kwargs.items():
        param[key] = value

    start = timer()
    stack1 = io.imread(stack1)
    stack2 = io.imread(stack2)
    end = timer()
    if verbose:
        print("Time it took to load images {}".format(end - start))

    assert stack1.shape == stack2.shape

    if frames == -1:
        frames = range(len(stack1))
    elif type(frames) == int:
        assert frames <= stack1.shape[0]
        frames = range(frames)

    os.makedirs(output_dir, exist_ok=True)

    output1 = os.path.join(output_dir, "warped1.tif")
    output2 = os.path.join(output_dir, "warped2.tif")

    motion_compensate(
        stack1, stack2, output1, output2, frames, param, verbose, parallel=False
    )


if __name__ == "__main__":
    main(
        "/mnt/internal_hdd/aymanns/181220_Rpr_R57C10_GC6s_tdTom/Fly5/002_coronal/2p/red.tif",
        "/mnt/internal_hdd/aymanns/181220_Rpr_R57C10_GC6s_tdTom/Fly5/002_coronal/2p/green.tif",
        "results/test_181220_Fly5_002_module",
        verbose=True,
        frames=2,
    )
