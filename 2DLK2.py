# New figure for LK^2

import numpy as np
# from numpy import cos, sin, exp 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from src.matrices import CovM
from src.covderivatives2D import AnalyticalSqExpdiff2D
# from calibration import adamopt, LOOMSESK2D, LOOVARSK2D
import Modified2DLagKrig as LK
# import Modified2DCoKrig as CK
from Modified2DCoKrig import Zpred, xpred, ypred, npred
# from Modified2DLagKrig import PredDerivatives
from numpy import exp, sin, cos

hypropt = 1                 # hyperparameter tuning
theta = 6.0
sigma = 1.0

Zobs = np.concatenate([LK.Zobs, 
                       LK.ExtZpred])    # Full vector of appended observations 

# Matrix K
K = CovM(Zobs, kernel = LK.DiffKernel, theta = theta)

# Matrix H
H = CovM(Zobs, Zpred, kernel = LK.DiffKernel, theta = theta)

# Values
yobs = LK.f1(LK.xfobs, LK.yfobs).flatten()
yderobs = LK.PredDerivatives(mode=1)       # Allpred from LK
ytrue = np.concatenate([yobs, yderobs])

# Final predictions
## Simple co-Kriging [yhat]
nugget = 1e-10
I = np.eye(K.shape[0]); nI = nugget * I
Kinv = np.linalg.solve(K + nI, I)
yhat = H.T @ Kinv @ ytrue 

if LK.f1flag == 1:
    skwargs = dict(marker = 'X', s = 100)
    plt.figure(figsize = (12,10))
    plt.cla()
    plt.contourf(xpred, ypred, yhat.reshape(npred, npred))
    # cnorm = Normalize(vmin = -1.0, vmax = 1.0)                     # To normalize the true values and scaling them to 0.0 --> 0.2 for the colorbar
    # plt.colorbar(cm.ScalarMappable(norm=cnorm), ax = plt.gca())
    # plt.clim(-1.0, 1.0)                                            # Same scale colorbar
    
    # cnorm = Normalize(vmin = -600.0, vmax = 600.0)                     # To normalize the true values and scaling them to 0.0 --> 0.2 for the colorbar
    # plt.colorbar(cm.ScalarMappable(norm=cnorm), ax = plt.gca())
    # plt.clim(-600.0, 600.0)                                            # Same scale colorbar

    plt.colorbar()
    plt.scatter(LK.xfobs, LK.yfobs, label = 'Observations', **skwargs, c = 'k')
    plt.scatter(LK.xL1, LK.yL1, label = 'Collocation', **skwargs, color = "#661100")
    plt.scatter(LK.xL2, LK.yL2, **skwargs, color = 'red')
    plt.title(f'$LK^2$ in 2D: {theta=:.2f}, {sigma=:.6f}')  
    # if label == 1: 
    #     plt.legend()
    # if figsave == 1:
    #     plt.savefig(IMGPATH+'CK2De1.pdf', dpi = 300, bbox_inches = 'tight', pad_inches = 0)
    #     plt.show()
    plt.show()


