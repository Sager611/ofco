import glob

import numpy as np

for f in glob.glob("/scratch/aymanns/200908_G23xU1/Fly2/001_coronal/w_*.npy"):
    w = np.load(f)
    print(w.shape)
