import sys
import os.path
import glob

from skimage import io
import numpy as np

import utils2p

from ofco import motion_compensate
from ofco.utils import default_parameters

def reassamble_warped_images(folder):
    n_files = len(glob.glob(os.path.join(folder, f"warped_red_*{tag}.tif")))
    stacks = []
    for i in range(n_files):
        substack = utils2p.load_img(os.path.join(folder, f"warped_red_{i}{tag}.tif"))
        stacks.append(substack)
    stack = np.concatenate(stacks, axis=0)
    utils2p.save_img(os.path.join(folder, f"warped_red{tag}.tif"), stack)

def reassamble_vector_fields(folder):
    n_files = len(glob.glob(os.path.join(folder, f"w_*{tag}.npy")))
    vector_fields = []
    for i in range(n_files):
        path = os.path.join(folder, f"w_{i}{tag}.npy")
        sub_fields = np.load(path)
        vector_fields.append(sub_fields)
    vector_field = np.concatenate(vector_fields, axis=0)
    np.save(os.path.join(folder, f"w{tag}.npy"), vector_field)


#folder = "/scratch/aymanns/200908_G23xU1/Fly2/004_coronal"
#ref_frame = io.imread("/scratch/aymanns/200908_G23xU1/Fly2/ref_frame.tif")
#folder = "/scratch/aymanns/200901_G23xU1/Fly1/001_coronal"
#ref_frame = io.imread("/scratch/aymanns/200901_G23xU1/Fly1/ref_frame.tif")
folder = sys.argv[1]
ref_frame = sys.argv[2]
print("Folder:", folder)
print("Reference frame:", ref_frame)
ref_frame = io.imread(ref_frame)

param = default_parameters()
#param["lmbd"] = 4000
tag = ""
for i, substack in enumerate(utils2p.load_stack_batches(os.path.join(folder, "red.tif"), 28)):
    print(i)
    frames = range(len(substack))
    warped_output = os.path.join(folder, f"warped_red_{i}{tag}.tif")
    w_output = os.path.join(folder, f"w_{i}{tag}.npy")
    if os.path.isfile(warped_output) and os.path.isfile(w_output):
        print("skipped because it exists")
        continue
    stack1_warped, stack2_warped = motion_compensate(
        substack, None, frames, param, parallel=True, verbose=True, w_output=w_output, ref_frame=ref_frame
    )

    io.imsave(warped_output, stack1_warped)
#io.imsave(os.path.join(folder, "warped2.tif"), stack2_warped)

reassamble_warped_images(folder)
reassamble_vector_fields(folder)
