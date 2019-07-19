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
import multiprocessing as mp
import array
from timeit import default_timer as timer
import numpy as np
from skimage import io
from docopt import docopt

from .utils import default_parameters, midway, crop_fit_size_center
from .warping import bilinear_interpolate
from .optflow import optical_flow_estimation


#class NpArrayManager(mp.BaseManager):
#    pass
#
#NpArrayManager.register('NpArray', np.array)


def compute_motion(I1, I2, param):
    sz0 = I1.shape

    I1 = np.pad(I1, [15, 15], "edge")
    I2 = np.pad(I2, [15, 15], "edge")

    # Optical flow
    #io.imsave('parallel_I1.tif',I1.astype(np.float32))
    #io.imsave('parallel_I2.tif',I2.astype(np.float32))
    #io.imsave('sequencial_I1.tif',I1.astype(np.float32))
    #io.imsave('sequencial_I2.tif',I2.astype(np.float32))
    w = optical_flow_estimation(I1, I2, sz0, param)

    w = crop_fit_size_center(w, [sz0[0], sz0[1], 2])
    return w


def parallel_compute_motion(t, m_array, w):
    i2 = global_stack1_rescale[t + 1, :, :].copy()
    [i10, i2] = midway(global_i1.copy(), i2)
    #return compute_motion(i10, i2, global_param)
    #io.imsave(f'parallel_i10_{t}.tif',i10.astype(np.float32))
    #io.imsave(f'parallel_i2_{t}.tif',i2.astype(np.float32))
    start = timer()
    #return_array[t] = compute_motion(i10, i2, global_param)
    res_w = compute_motion(i10, i2, global_param)
    end = timer()
    print(f'Time for {t} compute motion is {end - start} s')
    m_array.acquire()
    w[t] = res_w
    m_array.release()
    #w_size = w.shape[0] * w.shape[1] * w.shape[2]
    #return_array[t * w_size:(t + 1) * w_size] = array.array('d', w.flatten().tolist())


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
    w_shape = (
            stack1_rescale.shape[0] - 1,
            stack1_rescale.shape[1],
            stack1_rescale.shape[2],
            2,
    )

    i1 = stack1_rescale[frames[0], :, :]
    if parallel:
        global global_stack1_rescale
        global_stack1_rescale = stack1_rescale
        global global_i1
        global_i1 = i1
        global global_param
        global_param = param
        #with mp.Pool(2) as pool:
        #    output = pool.map(parallel_compute_motion, range(len(frames) - 1))
        #w = np.array(output)
        processes = []
        m_array = mp.Array('d', int(np.prod(w_shape)), lock=mp.Lock())
        w = np.frombuffer(m_array.get_obj(), dtype='d').reshape(w_shape)
        for i in range(len(frames) - 1):
            #p = mp.Process(target=parallel_compute_motion, args=(i, return_array))
            #p = mp.Process(target=parallel_compute_motion, args=(i, global_w))
            p = mp.Process(target=parallel_compute_motion, args=(i, m_array, w))
            processes.append(p)

        for i in range(1, len(processes), 2):
            processes[i - 1].start()
            processes[i].start()
            processes[i - 1].join()
            processes[i].join()
        if len(processes) % 2:
            processes[-1].start()
            processes[-1].join()

        #started = 0
        #joined = 0
        #while started < len(processes):
        #    if started - joined < 2:
        #        processes[started].start()
        #        started += 1
        #    if processes[joined].is_alive():


        #while any([proc.is_alive() for proc in processes]):
        #for p in processes:
        #    p.join()
    else:
        w = np.zeros(w_shape)
        for t in range(len(frames) - 1):
            if verbose:
                print("Frame {}\n".format(t))
            i2 = stack1_rescale[t + 1, :, :]
            [i10, i2] = midway(i1, i2)
            #io.imsave(f'sequencial_i10_{t}.tif',i10.astype(np.float32))
            #io.imsave(f'sequencial_i2_{t}.tif',i2.astype(np.float32))
            w[t, :, :, :] = compute_motion(i10, i2, param)
    #assert(np.allclose(w, global_w))
    #assert(np.allclose(w, w_parallel))
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
