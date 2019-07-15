from scipy.ndimage import gaussian_filter
import math
from skimage import transform
import numpy as np

from .utils import eigsDtD, partial_deriv, post_process

def of_l1_l2_fm_admm(I1, I2, sz0, param, verbose=False):
    
    sigmaPreproc = 0.9
    I1 = gaussian_filter(I1, sigmaPreproc, mode='mirror')
    I2 = gaussian_filter(I2, sigmaPreproc, mode='mirror')
    
    deriv_filter = np.array([[1, -8, 0, 8, -1]])/12.

    #coarse-to-fine parameters
    minSizeC2f = 10
    c2fLevels = int(math.ceil(math.log(minSizeC2f / max(I1.shape)) / math.log(1 / param['c2fSpacing'])))
    factor = math.sqrt(2)
    smooth_sigma = math.sqrt(param['c2fSpacing']) / factor
    I1C2f = reversed(list(transform.pyramid_gaussian(I1,
                                                     max_layer = c2fLevels - 1,
                                                     downscale = param['c2fSpacing'],
                                                     sigma = smooth_sigma,
                                                     multichannel=False)))
    I2C2f = reversed(list(transform.pyramid_gaussian(I2,
                                                     c2fLevels - 1,
                                                     param['c2fSpacing'],
                                                     smooth_sigma,
                                                     multichannel=False)))
   
    lambdaC2f = np.empty(c2fLevels)
    sigmaSSegC2f = np.empty(c2fLevels)
    gammaC2f = np.empty(c2fLevels)
    
    for i in range(c2fLevels - 1, -1, -1):
        lambdaC2f[i] = param['lmbd']
        sigmaSSegC2f[i] = param['sigmaS'] / (param['c2fSpacing'] ** i)
        gammaC2f[i] = param['gamma'] * ((c2fLevels - i) / c2fLevels) ** 1.8

    #Initationlization
    wl = np.zeros((I1.shape[0], I1.shape[1], 2))
    occ = np.ones(I1.shape)

    #Coarse to fine
    for l, I1, I2 in zip(range(c2fLevels - 1, -1, -1), I1C2f, I2C2f):
        if verbose:
            print('\nScale {}\n'.format(l))

        #Scaled data
        sigmaS = sigmaSSegC2f[l]
        lmbd = lambdaC2f[l]
        param['gamma'] = gammaC2f[l]

        #Resacle flow
        ratio = I1.shape[0] / wl[:,:,0].shape[0]
        ul = transform.resize(wl[:,:,0],I1.shape, order=3) * ratio
        ratio = I1.shape[1] / wl[:,:,1].shape[1]
        vl = transform.resize(wl[:,:,1],I1.shape, order=3) * ratio
        wl = np.dstack((ul,vl))
        sz0 = np.floor(np.array(sz0)*ratio)

        #Create binary and motion vectors fields
        c = np.zeros(I1.shape)
        m = np.zeros((I1.shape[0], I1.shape[1], 3))

        occ = transform.resize(occ, I1.shape, order=0)
        occ = occ.astype(np.bool_)

        mu = param['mu']
        nu = param['nu']

        eigs_DtD = eigsDtD(I1.shape[0], I1.shape[1], lmbd, mu)
        for iWarp in range(param['nbWarps']):
            if verbose:
                print('Warp {}\n'.format(iWarp))
            
            wPrev = wl
            dul = np.zeros((wl.shape[0], wl.shape[1]))
            dvl = np.zeros((wl.shape[0], wl.shape[1]))
            uu1 = np.zeros((wl.shape[0], wl.shape[1]))
            uu2 = np.zeros((wl.shape[0], wl.shape[1]))
            dwl = np.zeros((wl.shape))
            #ADMM variables
            alpha = np.zeros((I1.shape[0],I1.shape[1],2))
            beta = np.zeros((I1.shape[0],I1.shape[1],2))
            z = np.zeros((I1.shape[0],I1.shape[1],2))
            u = np.zeros((I1.shape[0],I1.shape[1],2))
            #No occlusion handling
            occ = np.ones(I1.shape)

            #Pre-computations 
            It, Ix, Iy = partial_deriv(np.stack([I1, I2], axis=2), wl, deriv_filter)
            
            Igrad = Ix**2 + Iy**2 + 1e-3
            idocc = (occ==0);

            #Main iterations loop
            for it in range(param['maxIters']):
                thresh = Igrad / (mu + nu)
                thresh2 = param['gamma'] * m[:, :, 2] / nu
                #Data update
                r1 = z - wl - alpha / mu
                r2 = u - wl + beta / nu
                t = (mu * r1 + nu * r2) / (mu + nu)
                t1 = t[:, :, 0]
                t2 = t[:, :, 1]

                rho = It + t1*Ix + t2*Iy
                idx1 = rho < -thresh
                idx2 = rho > thresh
                idx3 = np.abs(rho) <= thresh

                dul[idx1] = t1[idx1] + Ix[idx1] / [mu+nu]
                dvl[idx1] = t2[idx1] + Iy[idx1] / [mu+nu]

                dul[idx2] = t1[idx2] - Ix[idx2] / [mu+nu]
                dvl[idx2] = t2[idx2] - Iy[idx2] / [mu+nu]

                dul[idx3] = t1[idx3] - rho[idx3] * Ix[idx3] / Igrad[idx3]
                dvl[idx3] = t2[idx3] - rho[idx3] * Iy[idx3] / Igrad[idx3]

                dul[idocc] = t1[idocc]
                dvl[idocc] = t2[idocc]

                dwl = np.dstack((dul, dvl))
                
                #Regularization update
                z[:,:,0] = np.real(np.fft.ifft2(np.divide(np.fft.fft2(mu * (dwl[:,:,0] + wl[:,:,0]) + alpha[:,:,0]), eigs_DtD)))
                z[:,:,1] = np.real(np.fft.ifft2(np.divide(np.fft.fft2(mu * (dwl[:,:,1] + wl[:,:,1]) + alpha[:,:,1]), eigs_DtD)))

                #Matching update
                u0 = wl+dwl - beta/nu
                u01 = u0[:,:,0]
                u02 = u0[:,:,1]
                m1 = m[:,:,0]
                m2 = m[:,:,1]
                idx = (c == 0)

                rho = u0 - m[:,:,:2]
                idx1 = (rho[:,:,0]      < - thresh2) * (c == 1)
                idx2 = (rho[:,:,0]      >   thresh2) * (c == 1)
                idx3 = (abs(rho[:,:,0]) <=  thresh2) * (c == 1)

                uu1[idx1] = u01[idx1] + thresh2[idx1]
                uu1[idx2] = u01[idx2] - thresh2[idx2]
                uu1[idx3] = m1[idx3]
                uu1[idx] = u01[idx]

                idx1 = (rho[:,:,1]      < - thresh2) * (c == 1)
                idx2 = (rho[:,:,1]      >   thresh2) * (c == 1)
                idx3 = (abs(rho[:,:,1]) <=  thresh2) * (c == 1)

                uu2[idx1] = u02[idx1] + thresh2[idx1]
                uu2[idx2] = u02[idx2] - thresh2[idx2]
                uu2[idx3] = m2[idx3]
                uu2[idx] = u02[idx]

                u = np.dstack((uu1,uu2))
                
                #Lagrange parameters update
                alpha = alpha + mu*(wl+dwl - z)
                beta= beta + nu*(u-wl-dwl)
        
                #Post-processing
                if (it % param['iWM']) == 0:
                    w0 = wl + dwl
                    w0 = post_process(w0, I1, I2, sigmaS, param['sigmaC'])
                    dwl = w0 - wl
        
                #End of iterations checking
                w = wl + dwl
                if np.linalg.norm(wPrev.flatten()) == 0:
                    wPrev = w
                    continue
                else:
                    change = np.linalg.norm(w.flatten() - wPrev.flatten()) / np.linalg.norm(wPrev.flatten())
                    if change < param['changeTol']:
                        break
                    wPrev = w

            wl = wl + dwl
            wl = post_process(wl, I1, I2, sigmaS, param['sigmaC']);
    return wl
