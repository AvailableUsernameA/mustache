import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import skimage.io as io
import skimage.filters as flt
import numpy as np
import scipy.ndimage.filters as flt
import warnings
from PIL import Image
import copy
import pandas as pd

def anisodiff(img,niter=200,kappa=10,gamma=0.1):

    img = img.astype('float32')
    imgout = img.copy()

    deltaS = np.zeros_like(imgout)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(imgout)
    gE = gS.copy()

    for ii in np.arange(1,niter):

        deltaS[:-1,: ] = np.diff(imgout,axis=0)
        deltaE[: ,:-1] = np.diff(imgout,axis=1)

        deltaSf=deltaS;
        deltaEf=deltaE;

        gS = np.exp(-(deltaSf/kappa)**2.)
        gE = np.exp(-(deltaEf/kappa)**2.)
    
        E = gE*deltaE
        S = gS*deltaS

        NS[:] = S
        EW[:] = E
        NS[1:,:] -= S[:-1,:]
        EW[:,1:] -= E[:,:-1]

        # update the image
        imgout += gamma*(NS+EW)

    return imgout