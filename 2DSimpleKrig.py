# Simple Krging 2D scalar functions
# Imports
import numpy as np
from numpy import cos, sin, exp 
import matplotlib.pyplot as plt
from src.matrices import CovM
from src.covderivatives2D import AnalyticalSqExpdiff2D
from src.calibration import adamopt, LOOMSESK2D, LOOVARSK2D
import time

start = time.time()                                         # To compute the full run time
###################

# Constants
PI = np.pi
seed = 1
np.random.seed(seed)                                        # Reproducibility

# Choose function(s)
f1flag = 0;                                                 # To launch code for f(x, y) = cos(x) sin(y)
f2flag = 1;                                                 # To launch code for f(x, y) = e^x sin(y) + 5    
UQflag = 0;                                                 # time consuming to compute the UQ

# Scalar Functions
f1 = lambda X, Y: cos(X) * sin(Y)                           # f(x, y) = cos(x) sin(y)                                            
f2 = lambda X, Y: (exp(X) * sin(Y)) + 5                     # f(x, y) = e^x sin(y) + 5                  (Harmonic function)

# Utility functions
gridpoints = lambda xl, xr, yd, yu, n: np.meshgrid(np.linspace(xl, xr, int((n)**(0.5))), 
                                                   np.linspace(yd, yu, int((n)**(0.5))))

# user inputs
xl = 0; xr = 2 * PI
yd = 0; yu = 2 * PI
nobs = 10
nL1 = 50; nL2 = 100                                         # min. number of collocation points 4
disc = 50; npred = 30
sigma = 27.55; theta = 1.05                                 # f1opt = 1.74, 1.03; f2opt = 1.05, 27.55
IMGPATH = r'IMAGES/'                                        # Image path

# Flag settings
gridobs = 0;                                                # Grid observations
hypropt = 0;                                                # 1 => LOOCV based covariance paramaters calibration, 0 => Fixed kernel params (to check results with a certain param set)
label = 0;                                                  # 1 => To add a legend
figsave = 1;                                                # 1 => to save figs  

# Defining the geomtery
X = np.linspace(xl, xr, disc)
Y = np.linspace(yd, yu, disc)
X, Y = np.meshgrid(X, Y)
y1 = f1(X, Y); y2 = f2(X, Y)                                # True functions

xfobs = np.random.uniform(xl, xr, nobs)                     # Observation locations
yfobs = np.random.uniform(yd, yu, nobs)
if gridobs == 1:
    xfobs, yfobs = gridpoints(xl, xr, yd, yu, nobs)         # Grid observations <= nobs
    nobs = int(nobs**(0.5))**2
Zobs = np.zeros((nobs, 2, 2))                               # NUM X [x, y] X [m1, m2]
Zobs[:, 0, 0] = xfobs; Zobs[:, 0, 1] = yfobs                # Extended design space observations Z

xpred = np.linspace(xl, xr, npred)                          # Prediction locations
ypred = np.linspace(xl, xr, npred)
xpred, ypred = np.meshgrid(xpred, ypred)
Zpred = np.zeros((npred**2, 2, 2))                          # Derivative order is (0,0)
Zpred[:,0,0] = xpred.reshape(npred**2); Zpred[:,0,1] = ypred.reshape(npred**2) # Setting the locations

## Kernel differentiation funtion
DiffKernel = lambda x, y, xindex, yindex, **kwargs: AnalyticalSqExpdiff2D(x, y, 
                                                                          xindex, yindex, 
                                                                          **kwargs)

############################################
## LOOCV optimization steps
############################################
ytrue1, ytrue2 = np.zeros((2, nobs))            # Exact function observations
ytrue1 = f1(xfobs, yfobs).flatten();
ytrue2 = f2(xfobs, yfobs).flatten();

## Matrix K
def Kmat(params):
    CovParam = lambda *Zvals: CovM(*Zvals, kernel = DiffKernel, **params)
    K = CovParam(Zobs) 
    return K

if f1flag == 1:
    if hypropt == 1:
            bL = 0.5; bR = 1.0                                  # left and right bounds
            bounds = [bL, bR]                                   # To initialize theta (correlation parameter)
            n_iter = 1000
            ## f(x, y) = cos(x) sin(y)
            loomsefn1 = lambda theta: LOOMSESK2D(theta, ytrue1, Kmat, sigma = 1.0)
            thetaopt1, optscore1, history1 = adamopt(loomsefn1, bounds, n_iter = n_iter)
            print(f'{thetaopt1=:.2f}'); print(f'{optscore1=:.4f}')
            plt.plot(range(n_iter), history1); plt.show()      # To visualize LOOCV loss curve
            sigmaopt1 = LOOVARSK2D(ytrue1, Kmat, dict(theta = thetaopt1, sigma = 1.0))
            print(f'{sigmaopt1=:.2f}')
    else: 
            print(f'Using the fixed kernel params (user inputs): {theta=:.2f}, {sigma=:.2f}')
            thetaopt1 = theta; sigmaopt1 = sigma

    ## Matrix K
    ## f(x, y) = cos(x) sin(y)
    K = Kmat(dict(theta = thetaopt1,                 # LOOCV Optimal Kplus matrix 
                    sigma = 1.0))        
    
    ## Matrix H
    q = Zpred.shape[0]                                      # Number of prediction locations
    params1 = dict(theta = thetaopt1, sigma = 1.0)
    CovParam1 = lambda *Zvals: CovM(*Zvals, kernel = DiffKernel, **params1)
    H1 = CovParam1(Zobs, Zpred)
    
    # Final predictions
    ## Simple Kriging [yhat]
    nugget = 1e-10
    I = np.eye(K.shape[0]); nI = nugget * I
    Kinv1 = np.linalg.solve(K + nI, I)
    yhat1 = H1.T @ Kinv1 @ ytrue1 

    if UQflag == 1:
        ## Uncertainty quantification: simple co-Kriging expression
        BaseVar = 1e-5                                          # To make sure Conditional variance estimate is positive. 
        Kpred = CovParam1(Zpred)                                # Unconditional uncertainty 
        CondVar1 = Kpred - (H1.T @ Kinv1 @ H1);                 # Minimal MSE
        CondVar1 += BaseVar;                                    # Small quantity added to make it positive                         
        CondSigma1 = np.sqrt(sigmaopt1**2 * np.diag(CondVar1));

if f2flag == 1:
    if hypropt == 1:
            bL = 0.5; bR = 1.0                                  # left and right bounds
            bounds = [bL, bR]                                   # To initialize theta (correlation parameter)
            n_iter = 1000
            ## f(x, y) = e^x sin(y) + 5
            loomsefn2 = lambda theta: LOOMSESK2D(theta, ytrue2, Kmat, sigma = 1.0)
            thetaopt2, optscore2, history2 = adamopt(loomsefn2, bounds, n_iter = n_iter)
            print(f'{thetaopt2=:.2f}'); print(f'{optscore2=:.4f}')
            plt.plot(range(n_iter), history2); plt.show()      # To visualize LOOCV loss curve
            sigmaopt2 = LOOVARSK2D(ytrue2, Kmat, dict(theta = thetaopt2, sigma = 1.0))
            print(f'{sigmaopt2=:.2f}')
    else: 
            print(f'Using the fixed kernel params (user inputs): {theta=:.2f}, {sigma=:.2f}')
            thetaopt2 = theta; sigmaopt2 = sigma

    ## Matrix K
    ## f(x, y) = e^x sin(y) + 5
    K = Kmat(dict(theta = thetaopt2,                 # LOOCV Optimal Kplus matrix 
                    sigma = 1.0))        
    
    ## Matrix H
    q = Zpred.shape[0]                                      # Number of prediction locations
    params2 = dict(theta = thetaopt2, sigma = 1.0)
    CovParam2 = lambda *Zvals: CovM(*Zvals, kernel = DiffKernel, **params2)
    H2 = CovParam2(Zobs, Zpred)
    
    # Final predictions
    ## Simple Kriging [yhat]
    nugget = 1e-10
    I = np.eye(K.shape[0]); nI = nugget * I
    Kinv2 = np.linalg.solve(K + nI, I)
    yhat2 = H2.T @ Kinv2 @ ytrue2 

    if UQflag == 1:
        ## Uncertainty quantification: simple co-Kriging expression
        BaseVar = 1e-5                                          # To make sure Conditional variance estimate is positive. 
        Kpred = CovParam2(Zpred)                                # Unconditional uncertainty 
        CondVar2 = Kpred - (H2.T @ Kinv2 @ H2);                 # Minimal MSE
        CondVar2 += BaseVar;                                    # Small quantity added to make it positive                         
        CondSigma2 = np.sqrt(sigmaopt2**2 * np.diag(CondVar2));

# Visualization
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12})       
skwargs = dict(c = 'k', marker = 'X', s = 100)

if f1flag == 1:
    plt.figure(figsize = (12,10))
    plt.cla()
    plt.contourf(xpred, ypred, f1(xpred, ypred).reshape(npred, npred))
    plt.title(f'Exact function')  
    plt.colorbar()
    if figsave == 1:
        plt.savefig(IMGPATH+'Exacte1.pdf', dpi = 300, bbox_inches = 'tight', pad_inches = 0)

    plt.figure(figsize = (12,10))
    plt.cla()
    plt.contourf(xpred, ypred, yhat1.reshape(npred, npred))
    plt.clim(-1.0, 1.0)
    plt.colorbar()
    plt.scatter(xfobs, yfobs, label = 'Observations', **skwargs)
    plt.title(f'Simple Kriging 2D: {thetaopt1=:.2f}, {sigmaopt1=:.6f}')  
    if label == 1: 
        plt.legend()
    if figsave == 1:
        plt.savefig(IMGPATH+'SK2De1.pdf', dpi = 300, bbox_inches = 'tight', pad_inches = 0)
    plt.show()

    if UQflag == 1:
        plt.figure(figsize = (12,10))
        plt.cla()
        plt.contourf(xpred, ypred, CondSigma1.reshape(npred, npred))
        plt.colorbar()
        plt.scatter(xfobs, yfobs, label = 'Observations', **skwargs)
        plt.title(f'Simple Kriging UQ 2D: {thetaopt1=:.2f}, {sigmaopt1=:.6f}') 
        if label == 1: 
            plt.legend()
        if figsave == 1:
            plt.savefig(IMGPATH+'SK2DUQe1.pdf', dpi = 300, bbox_inches = 'tight', pad_inches = 0)
        plt.show()

if f2flag == 1:
    plt.figure(figsize = (12,10))
    plt.cla()
    plt.contourf(xpred, ypred, f2(xpred, ypred).reshape(npred, npred))
    plt.colorbar()
    plt.title(f'Exact function')  
    if figsave == 1:
        plt.savefig(IMGPATH+'Exacte2.pdf', dpi = 300, bbox_inches = 'tight', pad_inches = 0)

    plt.figure(figsize = (12,10))
    plt.cla()
    plt.contourf(xpred, ypred, yhat2.reshape(npred, npred))
    plt.clim(-600, 600)
    plt.colorbar()
    plt.scatter(xfobs, yfobs, label = 'Observations', **skwargs)
    plt.title(f'Simple Kriging 2D: {thetaopt2=:.2f}, {sigmaopt2=:.6f}')
    if label == 1: 
        plt.legend()
    if figsave == 1:
        plt.savefig(IMGPATH+'SK2De2.pdf', dpi = 300, bbox_inches = 'tight', pad_inches = 0)
    plt.show()
    
    if UQflag == 1:
        plt.figure(figsize = (12,10))
        plt.cla()
        plt.contourf(xpred, ypred, CondSigma2.reshape(npred, npred))
        plt.colorbar()
        plt.scatter(xfobs, yfobs, label = 'Observations', **skwargs)
        plt.title(f'Simple Kriging 2D: {thetaopt2=:.2f}, {sigmaopt2=:.6f}')
        if label == 1: 
            plt.legend()
        if figsave == 1:
            plt.savefig(IMGPATH+'SK2DUQe2.pdf', dpi = 300, bbox_inches = 'tight', pad_inches = 0)
        plt.show()

end = time.time()
#################
print(f'Program run time = {end - start:.2f}s')