# New experiment Fig 6.

import numpy as np
# from numpy import cos, sin, exp 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
# from matrices import CovM, Umat
# from covderivatives2D import AnalyticalSqExpdiff2D
# from calibration import adamopt, LOOMSESK2D, LOOVARSK2D
import Modified2DLagKrig as LK
import Modified2DCoKrig as CK
from Modified2DLagKrig import PredDerivatives
from Modified2DCoKrig import PredDerivativesCK

# Setting some important constants
LK.figsave = 0 
LK.UQFLAG = 0 
LK.hypropt = 0
LK.theta = 1.74

CK.figsave = 0
CK.UQflag = 0
CK.hypropt = 0
CK.theta = 1.41

FIGSAVE = 1                                                 # To save the output figure
IMGPATH = r'IMAGES/'                                        # Image path

LKyxhat, LKyyhat, LKyxxhat, LKyyyhat, SKyxhat, SKyyhat, SKyxxhat, SKyyyhat = PredDerivatives()
LKpredfns = [LKyxhat, LKyyhat, LKyxxhat, LKyyyhat]
SKpredfns = [SKyxhat, SKyyhat, SKyxxhat, SKyyyhat]
truefns = [LK.trueyx1, LK.trueyy1, LK.trueyxx1, LK.trueyyy1]
X, Y = LK.xL1, LK.yL1
shape = [int((LK.nL1)**(0.5)), int((LK.nL1)**(0.5))]

CKyxhat, CKyyhat, CKyxxhat, CKyyyhat = PredDerivativesCK()
CKpredfns = [CKyxhat, CKyyhat, CKyxxhat, CKyyyhat]

titles1 = [r'Actual $\partial_xf$', r'Actual $\partial_yf$', r'Actual $\partial^2_{xx}f$', r'Actual $\partial^2_{yy}f$']
titles3 = [r'LK $\partial_xf$', r'LK $\partial_yf$', r'LK $\partial^2_{xx}f$', r'LK $\partial^2_{yy}f$']
titles2 = [r'SK $\partial_xf$', r'SK $\partial_yf$', r'SK $\partial^2_{xx}f$', r'SK $\partial^2_{yy}f$']
titles4 = [r'CK $\partial_xf$', r'CK $\partial_yf$', r'CK $\partial^2_{xx}f$', r'CK $\partial^2_{yy}f$']

shapeCK1 = [int((CK.nqL1)**(0.5)), int((CK.nqL1)**(0.5))]
shapeCK2 = [int((CK.nqL2)**(0.5)), int((CK.nqL2)**(0.5))]
X1, Y1 = CK.xqL1, CK.yqL1
X2, Y2 = CK.xqL2, CK.yqL2  
shapeCK = [shapeCK1, shapeCK1, shapeCK2, shapeCK2]
XCK = [*(2*[X1]), *(2*[X2])]
YCK = [*(2*[Y1]), *(2*[Y2])]

# Visualize
fig, axs = plt.subplots(4, 4, figsize = (12,10), sharex=True, sharey=True)
plt.cla()
plt.suptitle(f'New Figure 6, LK theta = {LK.theta}, CK theta = {CK.theta}')
for _ in range(len(truefns)):
    c = axs[_,0].contourf(X, Y, truefns[_].reshape(*shape))
    c.set_clim(-1.0, 1.0)
    # plt.colorbar(c, ax = axs[_,0])
    axs[_,0].set_title(titles1[_])

    c = axs[_,1].contourf(X, Y, SKpredfns[_].reshape(*shape))
    c.set_clim(-1.0, 1.0)
    # plt.colorbar(c, ax = axs[_,1])s
    axs[_,1].set_title(titles2[_])

    c = axs[_,2].contourf(X, Y, LKpredfns[_].reshape(*shape))
    c.set_clim(-1.0, 1.0)
    # plt.colorbar(c, ax = axs[_,2])
    axs[_,2].set_title(titles3[_])

    c = axs[_,3].contourf(XCK[_], YCK[_], CKpredfns[_].reshape(*shapeCK[_]))
    c.set_clim(-1.0, 1.0)
    cnorm = Normalize(vmin = -1.0, vmax = 1.0)                     # To normalize the true values and scaling them to -1.0 --> 1.0 for the colorbar
    plt.colorbar(cm.ScalarMappable(norm=cnorm), ax = axs[_,3])
    axs[_,3].set_title(titles4[_])

if FIGSAVE == 1:
    plt.savefig(IMGPATH+'PredictingDerivatives-NewFigure-6.pdf', dpi = 300, bbox_inches = 'tight', pad_inches = 0)
plt.show()


