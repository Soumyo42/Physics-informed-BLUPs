# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: cfdvenvlocal
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Airfoil Co-Kriging
#
# Load the NACA airfoil coordinates and predict. The coordinates should be arranged in one sense. Meaning, that `XB` and `YB` are either in a clockwise or anti-clockwise arrangement. 
#
# ## Exact solution
#
# The true velocity is computed using the BEM code. 
# GRID discretization used:    
#
# ```{python}
# # Grid parameters
# nGridX   = 30                                                             # X-grid for streamlines and contours
# nGridY   = 30                                                             # Y-grid for streamlines and contours
# xVals    = [min(XB)-0.5, max(XB)+0.5]                                     # X-grid extents [min, max]
# yVals    = [min(YB)-0.2, max(YB)+0.2]                                     # Y-grid extents [min, max]
# ```

# %%
# imports 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from shapely.geometry import Polygon, Point
# from scipy.interpolate import griddata
from src.matrices import CovM, Umat
from src.covderivatives2D import AnalyticalSqExpdiff2D
from src.calibration import LOOMSECK2D, LOOVARCK2D, adamopt

# %%
# Image path to save the images.
IMGPATH = r'IMAGES/' 

# %%
# USER INPUTS
## Specify the airfoil code
NACAname = r'2424'
## Observation locations with derivative orders.
nobs = 20                                  # number of observations (at inf. on each side)
nm = 20                                    # number of observations in the middle of the domain.
nc = 5                                     # discretization for continuity collocation points.
nb = 30                                    # discretization for boundary collocation points. nb = 50 for NACA0012, nb = 30 for NACA4412, NACA2424.
UQFLAG = 0                                 # For uncertainty prediction
figsave = 0                                # To save the figures
## kernel parameters
hypropt = 0;                               # Flag to perform LOOCV hyperparameter tuning
sigma = 0.71;                               # Default noise variance
theta = 0.17;                              # Default lengthscale / correlation parameter.  NACA0012: 0.27, NACA2424: 0.2  

# load the NACA airfoil data
def loadGeometry(nobs=nobs, nm=nm, nc=nc, nb=nb):
    XB, YB, nX, nY = np.loadtxt('data/NACA'+NACAname+'UIUCnormalvecs.dat', 
                                unpack=True)
    XX, YY, Vx, Vy = np.loadtxt('data/NACA'+NACAname+'BEMflow.dat',
                                unpack=True)
    # %%
    # TEST : normal vectors visualization. 
    # plt.quiver(XX, YY, Vx, Vy)
    # plt.plot(XB, YB)
    # for _ in range(XB.shape[0]):
    #     plt.plot([XB[_], XB[_] + 0.05*nX[_]], 
    #              [YB[_], YB[_] + 0.05*nY[_]], c = 'red')
    # # plt.ylim(-0.75, 0.75)                     # Smaller scale makes the normal vectors more apparent.
    # plt.show()
    
    # %%
    # TEST : choice of observation locations
    # Vx[XX == min(XX)]
    # nLobs = 10;
    # nMobs = 2
    # step = int(30/nLobs)
    # plt.quiver(XX, YY, Vx, Vy)
    # plt.scatter(XX[XX==min(XX)][::step], YY[XX==min(XX)][::step],  
    #             s = 10, c = 'red', marker = 'x')
    # plt.scatter(XX[XX==max(XX)][::step], YY[XX==max(XX)][::step],  
    #             s = 10, c = 'red', marker = 'x')
    # plt.scatter(XX[YY==min(YY)][int(30/nMobs)::int(30/nMobs)], YY[YY==min(YY)][int(30/nMobs)::int(30/nMobs)],  
    #             s = 10, c = 'red', marker = 'x')
    # plt.scatter(XX[YY==max(YY)][int(30/nMobs)::int(30/nMobs)], YY[YY==max(YY)][int(30/nMobs)::int(30/nMobs)],  
    #             s = 10, c = 'red', marker = 'x')
    # plt.fill(XB, YB)

    # %%
    # TEST : obtain continuity collocation points
    # nc = 10;                                               # Discretization to choose collocation points.
    # airfoilshape = Polygon(np.column_stack([XB, YB]))      # Defining a polygon based on the airfoil profile.
    # xgrid = np.linspace(min(XX)-0.1,max(XX)+0.1,nc)                
    # ygrid = np.linspace(min(YY)-0.1,max(YY)+0.1,nc)
    # xgrid, ygrid = np.meshgrid(xgrid, ygrid)               # raw grid of points.
    # airfoilcontains = lambda x, y: airfoilshape.contains(Point(x, y))   # Function to check if point is within airfoil. 
    # withinairfoil = np.array(list(map(airfoilcontains, 
    #                                 xgrid.flatten(), 
    #                                 ygrid.flatten())))
    # Cx = xgrid.flatten()[~withinairfoil]       # retaining only the points outside the airfoil, x-coordinates
    # Cy = ygrid.flatten()[~withinairfoil]       # retaining only the points outside the airfoil, y-coordinates
    # plt.scatter(Cx, Cy, marker = 'x', s = 10)
    # plt.plot(XB, YB)
    # plt.show()

    # %% [markdown]
    # ## Defining relevant points in the extended design space.
    #  - Observation points (`Obs`). $v_x \rightarrow$ `m = (1,0)`. $v_y \rightarrow$ `m = (0,1)`.
    #  - Continuity collocation points (`C`). $\phi_{xx}\rightarrow$ `m = (2,0)`. $\phi_{yy}\rightarrow$ `m = (0,2)` 
    #  - Boundary collocation points (`B`). `m = (1,0)` \& `m = (0,1)`.
    #  - Prediction points (`Pred`). `m = (1,0)` \& `m = (0,1)`.
    #  

    ostep = int(np.sqrt(XX.shape[0])/nobs)     # step size to select observation points at infinity.
    mstep = int(np.sqrt(XX.shape[0])/nm)       # step size to select observation points in the domain.
    bstep = int(XB.shape[0]/nb)                # step size to select boundary points.

    # %%
    # Observation locations
    mask = np.zeros_like(XX, dtype = bool)      # mask to choose the right observation indices.
    
    for _ in np.where(XX==min(XX))[0][::ostep]:
        mask[_] = True;                         # observations at -infinity.
    for _ in np.where(XX==max(XX))[0][::ostep]:
        mask[_] = True;                         # observations at +infinity.
    for _ in np.where(YY==min(YY))[0][mstep::mstep]:
        mask[_] = True;                         # (upper) mid-domain observations (avoiding redundancy)
    for _ in np.where(YY==max(YY))[0][mstep::mstep]:
        mask[_] = True;                         # (lower) mid-domain observations (avoiding redundancy)
    
    # @Didier's test for increased observations. 
    # for _ in np.where(XX==np.sort(XX)[100])[0][::ostep]:
    #     mask[_] = True;                         # observations in the DOMAIN.
    # for _ in np.where(XX==np.sort(XX)[-100])[0][::ostep]:
    #     mask[_] = True;                         # observations in the DOMAIN.
    # for _ in np.where(XX==np.sort(XX)[50])[0][::ostep]:
    #     mask[_] = True;                         # observations in the DOMAIN.
    # for _ in np.where(XX==np.sort(XX)[-50])[0][::ostep]:
    #     mask[_] = True;                         # observations in the DOMAIN.
    # for _ in np.where(YY==np.sort(YY)[100])[0][::ostep]:
    #     mask[_] = True;                         # observations in the DOMAIN.
    # for _ in np.where(YY==np.sort(YY)[-100])[0][::ostep]:
    #     mask[_] = True;                         # observations in the DOMAIN.
    # for _ in np.where(YY==np.sort(YY)[50])[0][::ostep]:
    #     mask[_] = True;                         # observations in the DOMAIN.
    # for _ in np.where(YY==np.sort(YY)[-50])[0][::ostep]:
    #     mask[_] = True;                         # observations in the DOMAIN.
    # for _x in XX[::2]:
    #     for _y in YY[::2]:
    #         for _ in np.where((XX==_x)*(YY==_y))[0]:
    #             mask[_] = True
    # mask[::4] = True
    
    nobs = np.sum(mask)                         # redefining nobs based on the chosen locations.
    Obs = np.zeros((2*nobs, 2, 2))              # Observations of [Vx, Vy].
    
    ## Setting the appropriate locations. 
    Obs[:nobs, 0, 0] = XX[mask]; 
    Obs[:nobs, 0, 1] = YY[mask] 
    Obs[nobs:, 0, :] = Obs[:nobs, 0, :]         # repeating the same observation locations.
    
    ## Setting the right order of derivatives. 
    Obs[:nobs, 1, 0] = 1; Obs[nobs:, 1, 1] = 1  # m = (1,0) and m = (0,1) 

    # %% [markdown]
    # @David's experiment. Random observations instead of a uniform grid.
    #
    # Conclusion: Doesn't give that good results. The effect of few observations is that there is a lot of domain entry-exit behaviour (stochastic). Random or uniform grid barely makes a difference in that context. 
    
    # %%
    # To use random observations at infinity instead of a uniform gird of observations. 
    # Observation locations
    # mask = np.zeros_like(XX, dtype = bool)      # mask to choose the right observation indices.
    
    # for _ in np.random.choice(np.where(XX==min(XX))[0], nobs):
    #     mask[_] = True;                         # observations at -infinity.
    # for _ in np.random.choice(np.where(XX==max(XX))[0], nobs):
    #     mask[_] = True;                         # observations at +infinity.
    # for _ in np.random.choice(np.where(YY==min(YY))[0], nm):
    #     mask[_] = True;                         # (upper) mid-domain observations (avoiding redundancy)
    # for _ in np.random.choice(np.where(YY==max(YY))[0], nm):
    #     mask[_] = True;                         # (lower) mid-domain observations (avoiding redundancy)
    
    # nobs = np.sum(mask)                         # redefining nobs based on the chosen locations.
    # Obs = np.zeros((2*nobs, 2, 2))              # Observations of [Vx, Vy].
    
    # ## Setting the appropriate locations. 
    # Obs[:nobs, 0, 0] = XX[mask]; 
    # Obs[:nobs, 0, 1] = YY[mask] 
    # Obs[nobs:, 0, :] = Obs[:nobs, 0, :]         # repeating the same observation locations.
    
    # ## Setting the right order of derivatives. 
    # Obs[:nobs, 1, 0] = 1; Obs[nobs:, 1, 1] = 1  # m = (1,0) and m = (0,1) 

    # %%
    # TEST : Is the `mask` vector properly defined?
    # plt.scatter(XX[mask], YY[mask], marker = 'x')
    # plt.quiver(XX[mask], YY[mask],
    #            Vx[mask], Vy[mask])
    # plt.show() 

    # %% [markdown]
    # when choosing continuity collocation points, set the discretization based on the scale of the x-axis and y-axis. The discretization in our implemention differs by a factor of four which we found as, 
    # ```{python}
    # # setting the discretization to be representative of the scale on x-axis and y-axis
    # nc = 5                                                                      # A particular constant. Discretization on both axes is a multiple of `nc`.
    # rxy = (max(XX) - min(XX))/(max(YY) - min(YY))                               # ratio of the scale on x-axis and y-axis. 
    # nX = rxy * nc                                                               # WLOG, assuming the x-scale is higher. 
    # nY = nc         
    # ```

    # %%
    # continuity collocation points
    airfoilshape = Polygon(np.column_stack([XB, YB]))   # Defining a polygon based on the airfoil profile.
    # discretization on x and y axis is chosen as nY/nX = range(Y)/range(X) 
    xgrid = np.linspace(min(XX),max(XX),4*nc)           # collocation coordinates the prediction locations.              
    ygrid = np.linspace(min(YY),max(YY),nc)
    xgrid, ygrid = np.meshgrid(xgrid, ygrid)            # raw grid of points.
    airfoilcontains = lambda x, y: airfoilshape.contains(Point(x, y))   # Function to check if point is within airfoil. 
    withinairfoil = np.array(list(map(airfoilcontains, 
                                    xgrid.flatten(), 
                                    ygrid.flatten())))
    Cx = xgrid.flatten()[~withinairfoil]                # retaining only the points outside the airfoil, x-coordinates
    Cy = ygrid.flatten()[~withinairfoil]                # retaining only the points outside the airfoil, y-coordinates
    p1 = Cx.shape[0]                                    # number of collocation locations
    C = np.zeros((2*p1, 2, 2))                          # initializing the collocation points.
    
    # Setting the locations
    C[::2, 0, 0] = Cx; C[::2, 0, 1] = Cy                
    C[1::2, 0, :] = C[::2, 0, :]                        # locations repeating twice.
    
    # Setting the right order of derivatives.
    C[::2, 1, 0] = 2; C[1::2, 1, 1] = 2                 # m = (2,0) & m = (0,2)

    # %%
    # TEST : Are the continuity collocation points correct?
    # plt.scatter(C[::2, 0, 0], C[1::2, 0, 1], marker = 'x', s = 10)
    # plt.plot(XB, YB)
    # plt.show()

    # %%
    # boundary collocation points
    Bx = XB[::bstep]; By = YB[::bstep]                  # chosen boundary coordinates.
    p2 = Bx.shape[0]                                    # NUM of collocation points. 
    B = np.zeros((2*p2, 2, 2))                          # initialize: sparse subset of the boundary points.
    B[::2, 0, 0] = Bx; B[::2, 0, 1] = By
    B[1::2, 0, :] = B[::2, 0, :]                        # locations repeating twice.
    B[::2, 1, 0] = 1; B[1::2, 1, 1] = 1                 # m = (1,0) & m = (0,1)
    
    # normal vectors
    nvects = np.column_stack([nX[::bstep], nY[::bstep]])

    # %%
    # TEST : Are the boundary locations sensible? 
    # plt.scatter(B[::2, 0, 0], B[1::2, 0, 1], marker = 'x', s = 20, c = 'r')
    # plt.plot(XB, YB)
    # plt.show()
    
    # TEST : Are the normal vectors summing to 1.0
    # print(nvects[:,0]**2 + nvects[:,1]**2)

    # %%
    # prediction points
    q = XX.shape[0]                      # NUM prediction points.
    Pred = np.zeros((2*q, 2, 2))         # initialize the prediction locations. 
    
    # Setting the right locations.
    Pred[:q, 0, :] = np.column_stack([XX, YY])
    Pred[q:, 0, :] = Pred[:q, 0, :]
    
    # Setting the right order of derivatives. 
    Pred[:q, 1, 0] = 1                        # m = (1,0)
    Pred[q:, 1, 1] = 1                        # m = (0,1)
    
    # creating a mask to set predictions inside the airfoil to zero.
    withinairfoil = np.array(list(map(airfoilcontains, 
                                        XX, 
                                        YY)))
    PredOut = np.vstack([Pred[:q][~withinairfoil],              # vx outside airfoil
                         Pred[q:][~withinairfoil]])             # vy outside airfoil


    # %%
    # TEST : Are the prediction locations correct? 
    # plt.scatter(Pred[:q, 0, 0][~withinairfoil], 
    #             Pred[q:, 0, 1][~withinairfoil], marker = 'x')
    # plt.plot(XB, YB)
    # plt.show()

    # Exact observations
    v1, v2 = np.zeros(p1), np.zeros(p2)             # RHS of DiffEqns
    n = Obs.shape[0]                                # True velocity observations.
    ytrue = np.zeros(n + p1 + p2)                   # Full vector of observations.
    ytrue[:int(n/2)] = Vx[mask];                    # vx obs first and then vy obs.
    ytrue[int(n/2):n] = Vy[mask];
    ytrue[n:n+p1] = v1                              # Laplacian = 0
    ytrue[n+p1:] = v2                               # normal component  = 0

    # Matrix U (coefficients for both differential equations)
    U1 = Umat([1, 1], p1)                           # Laplacian coefficients. 
    U2 = Umat(nvects, p2)                           # normal vector components. 

    # %% [markdown]
    # ## Kernel hyperparameters and covariance matrices
    # Set the variables `sigma` and `theta` to construct the covariance matrices OR opt for LOOCV hyperparameter tuning by setting `hypropt=1`.
    
    # Matrix K^+
    def Kmatfn(params):
        CovParam = lambda *Z: CovM(*Z, kernel=AnalyticalSqExpdiff2D, **params)
        K11 = CovParam(Obs);                        # Block matrix structure
        K12 = CovParam(Obs, C) @ U1; K21 = K12.T                 
        K13 = CovParam(Obs, B) @ U2; K31 = K13.T
        K22 = U1.T @ CovParam(C) @ U1
        K23 = U1.T @ CovParam(C, B) @ U2; K32 = K23.T
        K33 = U2.T @ CovParam(B) @ U2
        Kplus = np.block([[K11, K12, K13],
                          [K21, K22, K23],
                          [K31, K32, K33]])
        return Kplus
    return  XX, YY, XB, YB, Vx, Vy, mask, Obs, C, B, PredOut, Pred, withinairfoil, ytrue, U1, U2, n, q, Kmatfn   

# Define all required variables.
XX, YY, XB, YB, Vx, Vy, mask, Obs, C, B, PredOut, Pred, withinairfoil, ytrue, U1, U2, n, q, Kmatfn = loadGeometry()

# %%
if __name__ == '__main__': 

    ##########################
    # LOOCV optimization step
    ##########################
    if hypropt == 1:
        bL = 0.1; bR = 0.2                                  # left and right bounds
        bounds = [bL, bR]                                   # To initialize theta (correlation parameter)
        n_iter = 100
        loomsefn = lambda theta: LOOMSECK2D(theta, ytrue, n, Kmatfn, sigma = 1.0)
        thetaopt, optscore, history = adamopt(loomsefn, bounds, n_iter = n_iter, stoploss = 50, alpha = 1e-3)
        print(f'{thetaopt=}'); print(f'{optscore=:.4f}')
        sigmaopt = LOOVARCK2D(ytrue, n, Kmatfn, dict(theta = thetaopt,
                                                    sigma = 1.0), nugget = 1e-4)        # Big nugget for numerical stability of the (K^+)^{-1}
        print(f'{sigmaopt=:.2f}')
        plt.figure()
        plt.plot(range(n_iter), history); plt.title(f'Loss curve with {thetaopt=}, {sigmaopt=:.2f}')
        plt.savefig(IMGPATH + NACAname +'lossAirfoilCK.pdf')
        # plt.show()      # To visualize LOOCV loss curve
    else: 
        print(f'Using the fixed kernel params (user inputs): {theta=}, {sigma=:.2f}')
        thetaopt = theta; 
        sigmaopt = sigma
        # sigmaopt = LOOVARCK2D(ytrue, n, Kmatfn, dict(theta = thetaopt,
        #                                             sigma = 1.0), nugget = 1e-6)
    ## Matrix K^+ (optimal)
    Kplus = Kmatfn(dict(theta = thetaopt,                       # LOOCV Optimal Kplus matrix 
                         sigma = 1.0)) 
    
    # %% [markdown]
    # ## Final predictions and UQ
    
    # %%
    ## Matrix H^+
    params = dict(theta = thetaopt, sigma = 1.0)
    CovParam = lambda *Zvals: CovM(*Zvals, kernel = AnalyticalSqExpdiff2D, **params)
    H1 = CovParam(Obs, PredOut)
    H2 = U1.T @ CovParam(C, PredOut)
    H3 = U2.T @ CovParam(B, PredOut)
    Hplus = np.block([[H1], 
                      [H2], 
                      [H3]])

    # %%
    # Simple co-Kriging [yhat]
    # yhat = np.zeros(2*q)                                        # Prediction of [Vx, Vy]
    qout = int(PredOut.shape[0]/2)
    nugget = 1e-2                                                 # small nugget for numerical stability.
    I = np.eye(Kplus.shape[0]); nI = nugget * I
    Kplusinv = np.linalg.solve(Kplus + nI, I)
    vxhat, vyhat = (Hplus.T @ Kplusinv @ ytrue).reshape(2, qout)
    # yhat = Hplus.T @ Kplusinv @ ytrue

    # %%
    ## Visualization true flow
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12})    
    widf = 1.5
    quiverkwargs = dict(pivot = 'tail',
                        scale = 1/0.030, 
                        width = 3.25/(widf*1e3), 
                        clim = (0.6, 1.1))              # Colorbar
    # %%
    plt.figure(figsize = (12, 10))
    plt.cla()
                                  
    Q1 = plt.quiver(XX, YY,
                    Vx, Vy,
                    np.hypot(Vx, Vy), 
                    alpha = 1,
                    **quiverkwargs) 
    plt.colorbar()
    plt.plot(XB, YB,
             alpha = 0.5, c = 'grey', label='Obstacle')
    plt.title(f'Exact flow')
    plt.xlim(min(XX)-0.05, max(XX)+0.05)
    plt.ylim(min(YY)-0.01, max(YY)+0.01)
    if figsave == 1:
        plt.savefig(IMGPATH+NACAname+'ExactAirfoil.pdf', dpi = 300, bbox_inches = 'tight', pad_inches = 0)
    else:
        plt.show()
    
    # %%
    ## Visualization Co-Kriging flow prediction
    plt.figure(figsize = (12, 10))
    plt.cla()
    # Q2 = plt.quiver(PredOut[:qout, 0, 0], PredOut[:qout, 0, 1],
    #                 yhat[:qout], yhat[qout:],
    #                 np.hypot(yhat[:qout], yhat[qout:]), 
    #                 alpha = 1.0,
    #                 **quiverkwargs)
    Q2 = plt.quiver(PredOut[:qout, 0, 0], PredOut[:qout, 0, 1],
                    vxhat, vyhat,
                    np.hypot(vxhat, vyhat), 
                    alpha = 1.0,
                    **quiverkwargs)
    plt.colorbar()
    Q1 = plt.quiver(XX[mask], YY[mask],
                    Vx[mask], Vy[mask],
                    # np.hypot(vwx[mask], vwy[mask]), 
                    # alpha = 0.5,
                    color = 'red', alpha = 1,
                    **quiverkwargs) 
    plt.quiverkey(Q1, 1.02, 1.05, 1, label='Observations', labelpos = 'E')
    plt.quiverkey(Q2, 1.02, 1.03, 1, label='Predictions', labelpos = 'E')
    plt.plot(XB, YB,
             alpha = 0.5, c = 'grey', label='Obstacle')
    plt.title(f'Co-Kriging: {thetaopt=}, {sigmaopt=:.4f}')
    plt.xlim(min(XX)-0.05, max(XX)+0.05)
    plt.ylim(min(YY)-0.01, max(YY)+0.01)
    if figsave == 1:
        plt.savefig(IMGPATH+NACAname+'CKAirfoil.pdf', dpi = 300, bbox_inches = 'tight', pad_inches = 0)
    else:
        plt.show()

    ## Visualization flow prediction diff.
    # plt.figure(figsize = (12, 10))
    # plt.cla()
    # Q2 = plt.quiver(PredOut[:qout, 0, 0], PredOut[:qout, 0, 1],
    #                 vxhat - Vx[~withinairfoil], vyhat - Vy[~withinairfoil],
    #                 np.hypot(vxhat - Vx[~withinairfoil],
    #                         vyhat - Vy[~withinairfoil]), 
    #                 alpha = 1.0,
    #                 **quiverkwargs)
    # plt.colorbar()
    # plt.quiverkey(Q2, 1.02, 1.03, 1, label='Prediction error', labelpos = 'E')
    # plt.plot(XB, YB,
    #          alpha = 0.5, c = 'grey', label='Obstacle')
    # plt.title(f'Co-Kriging error: {thetaopt=}, {sigmaopt=:.4f}')
    # plt.xlim(min(XX)-0.05, max(XX)+0.05)
    # plt.ylim(min(YY)-0.05, max(YY)+0.05)
    # if figsave == 1:
    #     plt.savefig(IMGPATH+NACAname+'CKAirfoil.pdf', dpi = 300, bbox_inches = 'tight', pad_inches = 0)
    # else:
    # plt.show()
    
    
    # %% [markdown]
    # ## Uncertainty quantification.

    # %%
    if UQFLAG == 1:
        ## Uncertainty quantification: simple co-Kriging expression
        BaseVar = 1e-5                                       # To make sure Conditional variance estimate is positive. 
        Kpred = CovParam(PredOut)                            # Unconditional variance
        CondVar = Kpred - (Hplus.T @ Kplusinv @ Hplus)       # Minimal MSE
        CondVar += BaseVar                                   # Small quantity added to make it positive
        CondSigma = np.sqrt(sigmaopt**2 * np.diag(CondVar));
    
    # %%
    ## Co-Kriging uncertainty in prediction.
    ## Generalized chi^2 distribtion. Quadratic form of two dependent gaussian vectors vx, vy. 
    if UQFLAG == 1:
        CondVar = sigmaopt**2 * CondVar;
        CVxVx = np.diag(CondVar[:qout, :qout])                        # Var[Vx]
        CVxVy = np.diag(CondVar[:qout, qout:])                        # Cov[Vx, Vy]
        CVyVy = np.diag(CondVar[qout:, qout:])                        # Var[Vy]
        ChiVar = np.zeros(qout)                                       # Chi^2 params.
        for i in range(qout):
            Mu = np.array([vxhat[i],                                  # predicted Vx 
                           vyhat[i]])                                 # predicted Vy
            Cov = np.array([[CVxVx[i], CVxVy[i]],                     # Cov matrix for point prediction
                            [CVxVy[i], CVyVy[i]]])
            eigvect, eigmatrix = np.linalg.eigh(Cov)
            b = eigmatrix @ Mu
            ChiVar[i] = np.sum(2 * eigvect**2 * (1 + 2*b**2))         # A derivation can be found in https://math.stackexchange.com/a/442916
                                                                      # A paper about normal ---> generalized chi^2: https://arxiv.org/pdf/2012.14331
        plt.figure(figsize = (12,10))
        plt.cla()
        UQ = np.zeros(XX.shape[0])                                    # initializing the uncertainty array.
        UQ[~withinairfoil] = ChiVar                                   # Assigning the computed uncertainty to points outside the airfoil.
        plt.contourf(XX.reshape(30, 30), YY.reshape(30,30), 
                     2*np.sqrt(UQ).reshape(30, 30))
                    #  vmin = 0.0, vmax = 1.0)                        # Same scale colorbar
        cnorm = Normalize(vmin = 0.0, vmax = 2.0)                     # To normalize the true values and scaling them to 0.0 --> 0.2 for the colorbar
        plt.colorbar(cm.ScalarMappable(norm=cnorm), ax = plt.gca())
        plt.clim(0.0, 2.0)                                            # Same scale colorbar
        # plt.colorbar()
        plt.fill(XB, YB,                                              
                alpha = 1, c = 'white', label='Obstacle')
        plt.title(f'Co-Kriging UQ: {thetaopt=}, {sigmaopt=:.4f}')
        if figsave == 1:
            plt.savefig(IMGPATH+NACAname+'AirfoilCKUQ.pdf', dpi = 300, bbox_inches = 'tight', pad_inches = 0)
        else:
            plt.show()

        
#%%
## Some useful functions that can be called by other scripts ##
def CKpredictKHy():
    # kernel parameters
    thetaopt = theta; 

    ## Matrix K^+ 
    Kplus = Kmatfn(dict(theta = thetaopt,                       # Kplus matrix 
                         sigma = 1.0)) 
    
    ## Matrix H^+
    params = dict(theta = thetaopt, sigma = 1.0)
    CovParam = lambda *Zvals: CovM(*Zvals, kernel = AnalyticalSqExpdiff2D, **params)
    PredOut = np.vstack([Pred[:q][~withinairfoil],              # vx outside airfoil
                         Pred[q:][~withinairfoil]])             # vy outside airfoil
    H1 = CovParam(Obs, PredOut)
    H2 = U1.T @ CovParam(C, PredOut)
    H3 = U2.T @ CovParam(B, PredOut)
    Hplus = np.block([[H1], 
                      [H2], 
                      [H3]])
    return Kplus, Hplus, ytrue

def CKRelErr(Kplus, Hplus, ytrue, nugget):
    # Simple co-Kriging [yhat]
    qout = int(PredOut.shape[0]/2)
    I = np.eye(Kplus.shape[0]); nI = nugget * I         # small nugget for numerical stability.
    Kplusinv = np.linalg.solve(Kplus + nI, I)
    N = Kplusinv.shape[0]

    # LOOCV error 
    def loocvmse():
        GAMMA = Kplusinv @ ytrue
        DIAGK = np.diag((np.ones(N)/np.diag(Kplusinv))**2)
        # Sieve out the true observations ONLY.
        DIAG = np.zeros((N,N))
        DIAG[:n, :n] = np.eye(n) 
        loomse = 1/n * (GAMMA.T @ DIAGK @ DIAG @ GAMMA)
        return loomse
    
    LoocvErr = loocvmse()
    
    # Debugging: If loocv less than 

    # prediction error
    vxhat, vyhat = (Hplus.T @ Kplusinv @ ytrue).reshape(2, qout)
    Vxhat = np.zeros_like(Vx); Vyhat = np.zeros_like(Vy)
    Vxhat[~withinairfoil] = vxhat;
    Vyhat[~withinairfoil] = vyhat;

    # TEST : Are these the right vectors?
    # plt.quiver(XX, YY,
    #             Vx, Vy,
    #             color = 'red', alpha = 1,
    #             **quiverkwargs)
    # plt.title('Exact vectors')
    # plt.show() 

    # plt.quiver(XX, YY,
    #             Vxhat, Vyhat,
    #             color = 'red', alpha = 1,
    #             **quiverkwargs) 
    # plt.title('Predicted vectors')
    # plt.show()

    # plt.quiver(XX, YY,
    #             Vx - Vxhat, Vy - Vyhat,
    #             color = 'red', alpha = 1,
    #             **quiverkwargs)
    # plt.title('Difference')
    # plt.show() 

    # plt.scatter(XX, YY, c = np.hypot(Vx, Vy))
    # plt.colorbar()
    # plt.title('Exact magintude')
    # plt.show()

    # plt.scatter(XX, YY, c = np.hypot(Vxhat - Vx, Vyhat - Vy))
    # plt.colorbar()
    # plt.title('Difference magnitude')
    # plt.show()

    # diffmag = np.hypot(Vxhat - Vx, Vyhat - Vy)[~withinairfoil]
    # actmag = np.hypot(Vx, Vy)[~withinairfoil]
    # plt.scatter(XX[~withinairfoil], YY[~withinairfoil], 
    #             c = np.clip(diffmag/(actmag+1e-10), 0, 1))
    # plt.colorbar()
    # plt.show()

    AbsErr = np.mean((np.hypot(Vxhat - Vx, Vyhat - Vy)[~withinairfoil]))
    RelErr = 100*np.mean(np.clip(AbsErr/(np.hypot(Vx, Vy)[~withinairfoil]+1e-10),0,1))

    return RelErr, LoocvErr, AbsErr