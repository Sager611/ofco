"""
ofco

An optical flow-based motion correction algorithm for 2-photon calcium images.

Usage:
    ofco [-v | --verbose] [--frames=<int>] <stack1> <stack2> <output>
    ofco -h | --help

Options:
    -h --help       Show this screen.
    -v --verbose    Verbose output.
    --frames=<int>        Number of frames to correct [default: all].
"""

import os
import time
import logging
import multiprocessing as mp
from concurrent.futures.thread import ThreadPoolExecutor
from timeit import default_timer as timer
import numpy as np
from skimage import io
from docopt import docopt

from .utils import default_parameters, midway, crop_fit_size_center, get_strategy
from .warping import bilinear_interpolate
from .optflow import optical_flow_estimation

_LOGGER = logging.getLogger('ofco')


def compute_motion(I1, I2, param, initial_w=None, verbose=False):
    sz0 = I1.shape

    I1 = np.pad(I1, [param["padding"], param["padding"]], "edge")
    I2 = np.pad(I2, [param["padding"], param["padding"]], "edge")
    if initial_w is not None:
        initial_w = np.pad(
            initial_w,
            [
                (param["padding"], param["padding"]),
                (param["padding"], param["padding"]),
                (0, 0),
            ],
            mode="constant",
            constant_values=0,
        )

    # Optical flow
    w = optical_flow_estimation(I1, I2, sz0, param, initial_w=initial_w, verbose=verbose)

    w = crop_fit_size_center(w, [sz0[0], sz0[1], 2])
    return w


def apply_motion_field(stack1, stack2, w, frames):

    stack1_warped = stack1

    if stack2 is not None:
        stack2_warped = stack2
    else:
        stack2_warped = None

    for t in range(len(frames) - 1):
        stack1_warped[t + 1, :, :] = bilinear_interpolate(
            stack1[t + 1, :, :], w[t, :, :, 0], w[t, :, :, 1]
        )
        if stack2 is not None:
            stack2_warped[t + 1, :, :] = bilinear_interpolate(
                stack2[t + 1, :, :], w[t, :, :, 0], w[t, :, :, 1]
            )
    return stack1_warped, stack2_warped


def motion_compensate(
    stack1,
    stack2,
    frames,
    param,
    verbose=False,
    parallel=True,
    w_output=None,
    initial_w=None,
    ref_frame=None,
):
    if initial_w is not None and len(frames) - 1 != len(initial_w):
        raise ValueError(
            "Number of frames does not match the number of displacement vector fields provided in initial_w."
        )

    start = timer()
    stack1 = stack1[frames, :, :]
    if stack2 is not None:
        stack2 = stack2[frames, :, :]
    if ref_frame is not None:
        stack1 = np.concatenate((ref_frame[np.newaxis], stack1), axis=0)
        if stack2 is not None: 
            stack2 = np.concatenate((ref_frame[np.newaxis], stack2), axis=0)
        frames = (0,) + tuple(frames)
    stack1_rescale = (
        (stack1 - np.amin(stack1)) / (np.amax(stack1) - np.amin(stack1)) * 255
    )
    end = timer()
    if verbose:
        _LOGGER.info("Time it took to normalize images {}".format(end - start))

    # Motion estimation
    start = timer()
    w_shape = (
        stack1_rescale.shape[0] - 1,
        stack1_rescale.shape[1],
        stack1_rescale.shape[2],
        2,
    )

    i1 = stack1_rescale[frames[0], :, :]
    with get_strategy('GPU').scope():
        if parallel:
            #############################################################
            # Adrian Sager 15/11/2021:
            #     Changed from program-parallel to thread-parallel
            #
            #############################################################
            ## THREAD-LEVEL
            def parallel_compute_motion(t, frame_rescale):
                t1 = time.perf_counter()
                i2 = frame_rescale
                [i10, i2] = midway(i1, i2)
                if initial_w is not None:
                    res = compute_motion(i10, i2, param, initial_w[t], verbose=verbose)
                else:
                    res = compute_motion(i10, i2, param, verbose=verbose)
                t2 = time.perf_counter()
                if verbose:
                    _LOGGER.info(f'COMPUTED FRAME {t} IN {t2-t1 : .2f}s \n')
                return res

            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(parallel_compute_motion,
                                    t, stack1_rescale[t + 1, :, :])
                    for t in range(len(frames) - 1)
                ]
                w = np.array([f.result() for f in futures])
            
            ## PROCESS-PARALLEL
            # def parallel_compute_motion(t):
            #     t1 = time.perf_counter()
            #     i2 = global_stack1_rescale[t + 1, :, :].copy()
            #     [i10, i2] = midway(global_i1.copy(), i2)
            #     if global_initial_w is not None:
            #         res = compute_motion(i10, i2, global_param, global_initial_w[t])
            #     else:
            #         res = compute_motion(i10, i2, global_param)
            #     t2 = time.perf_counter()
            #     if verbose:
            #         _LOGGER.info(f'COMPUTED FRAME {t} IN {t2-t1 : .2f}s \n')
            #     return res

            # _LOGGER.info('PROCESS-LEVEL')
            # global global_stack1_rescale
            # global_stack1_rescale = stack1_rescale
            # global global_i1
            # global_i1 = i1
            # global global_param
            # global_param = param
            # global global_initial_w
            # global_initial_w = initial_w
            # with mp.Pool(28) as pool:
            #     w = pool.map(parallel_compute_motion, range(len(frames) - 1))
            # w = np.array(w)
            # del global_stack1_rescale
            # del global_i1
            # del global_param
        else:
            w = np.zeros(w_shape)
            for t in range(len(frames) - 1):
                if verbose:
                    print("Frame {}\n".format(t))
                i2 = stack1_rescale[t + 1, :, :]
                [i10, i2] = midway(i1, i2)
                if initial_w is not None:
                    w[t, :, :, :] = compute_motion(i10, i2, param, initial_w[t], verbose=verbose)
                else:
                    w[t, :, :, :] = compute_motion(i10, i2, param, verbose=verbose)
            del i2
            del i10
    end = timer()
    if verbose:
        print("Time it took to compute motion field w {}".format(end - start))

    del i1
    del stack1_rescale

    start = timer()
    stack1_warped, stack2_warped = apply_motion_field(stack1, stack2, w, frames)
    end = timer()
    if verbose:
        print("Time it took to warp images {}".format(end - start))

    if ref_frame is not None:
        stack1_warped = stack1_warped[1:]
        if stack2_warped is not None:
            stack2_warped = stack2_warped[1:]

    if w_output is not None:
        np.save(w_output, w)

    return stack1_warped, stack2_warped


def main(stack1, stack2, output_dir, frames=-1, verbose=False, **kwargs):
    """
    Parameters
    ----------
    stack1 : string
        Path to stack with constant brightness (e.g. tdTom).
    stack2 : string or None
        Path to stack with functional information (e.g. GCamP).
        If None, None is returned instead of a warped image.
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

    stack1_warped, stack2_warped = motion_compensate(
        stack1, stack2, frames, param, verbose, parallel=False
    )

    start = timer()
    io.imsave(output1, stack1_warped.astype(np.float32))
    io.imsave(output2, stack2_warped.astype(np.float32))
    end = timer()
    if verbose:
        print("Time it took to save images {}".format(end - start))


def cli():
    args = docopt(__doc__)
    if args["--frames"] == "all":
        frames = -1
    else:
        frames = int(args["--frames"])
    main(
        args["<stack1>"],
        args["<stack2>"],
        args["<output>"],
        verbose=args["--verbose"],
        frames=frames,
    )


if __name__ == "__main__":
    args = docopt(__doc__)
    if args["--frames"] == "all":
        frames = -1
    else:
        frames = int(args["--frames"])
    main(
        args["<stack1>"],
        args["<stack2>"],
        args["<output>"],
        verbose=args["--verbose"],
        frames=frames,
    )
