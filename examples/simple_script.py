from skimage import io
from ofco import motion_compensate
from ofco.utils import default_parameters

stack1 = io.imread(
    "/mnt/internal_hdd/aymanns/181220_Rpr_R57C10_GC6s_tdTom/Fly5/002_coronal/2p/red.tif"
)
stack2 = io.imread(
    "/mnt/internal_hdd/aymanns/181220_Rpr_R57C10_GC6s_tdTom/Fly5/002_coronal/2p/green.tif"
)

param = default_parameters()
frames = range(20)

motion_compensate(
    stack1,
    stack2,
    "warped1.tif",
    "warped2.tif",
    frames,
    param,
    parallel=True,
    verbose=True,
)
