from skimage import io
import numpy as np

from ofco import motion_compensate
from ofco.utils import default_parameters

stack1 = io.imread(
    "/mnt/internal_hdd/aymanns/181220_Rpr_R57C10_GC6s_tdTom/Fly5/002_coronal/2p/red.tif"
)
stack2 = io.imread(
    "/mnt/internal_hdd/aymanns/181220_Rpr_R57C10_GC6s_tdTom/Fly5/002_coronal/2p/green.tif"
)

param = default_parameters()
frames = range(len(stack1))

stack1_warped, stack2_warped = motion_compensate(
    stack1, stack2, frames, param, parallel=True, verbose=True
)

io.imsave("warped1.tif", stack1_warped)
io.imsave("warped2.tif", stack2_warped)
