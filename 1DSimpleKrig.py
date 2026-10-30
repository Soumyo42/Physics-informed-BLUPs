# 1D Simple Kriging
# Imports
import numpy as np
import time
import matplotlib.pyplot as plt
from src.matrices import CovM
from src.calibration import adamopt, LOOMSESK1D, LOOVARSK1D

# Constants
PI = np.pi
seed = 20                                               # Reproducibility 
np.random.seed(seed)

# Functional definitions
## Utility functions
printline = lambda *string: print(*string, '\n', 100 * '-' , '\n' , 100 * '-')  
idx_dict = lambda xindex, yindex: dict(xindex = int(xindex), yindex = int(yindex))

## Unknown function and ODE: f+f'' = 0
f = lambda X: np.sin(X)
fx = lambda X: np.cos(X)
fxx = lambda X: -np.sin(X)

# Inputs
disc: int = 100                                         # Discretization of the input space
xl: float = 0.0                                         # Lower limit
xr: float = 2 * PI                                      # in radians
limits = (xl, xr)  
X = np.linspace(*limits, disc)                          # input array
nfobs: int = 4                                          # Observations of the actual function
nfpred: int = 10                                        # prediction points, [= disc] for whole domain    
sigma: float = 0.61                                     # Noise variance parameter
theta: float = 0.83                                     # Correlation parameter
IMGPATH = r'IMAGES/'                                    # Image path

# Flag settings
paperobs = 1;                                           # To use the same observation locations as the paper.
bdryobs = 1;                                            # 1 => To make the problem well posed, add one boundary observation, 0 => no boundary condition
hypropt = 0;                                            # 1 => LOOCV based covariance paramaters calibration, 0 => Fixed kernel params (to check results with a certain param set)
label = 0;                                              # 1 => To add a legend
figsave = 1;                                            # 1 => to save figs    

# Creating important arrays
xfobs = np.random.uniform(*limits, nfobs)               # Randomly selected observation locations of f
if paperobs == 1:
    xfobs = np.array([0., 2.35330497,                   # Original code observations
                      4.59925358, 5.97351416])  
elif bdryobs == 1:                                      # Enables one boundary observation location
    xfobs = np.zeros(nfobs)
    xfobs[1:] = np.random.uniform(*limits, nfobs - 1)                
xfobs.sort(); xfpred = np.linspace(*limits, nfpred)     # Prediction locations of f

Zobs = np.zeros((xfobs.shape[0], 2))                    # Extended design space: Z = (fval, m)
Zpred = np.zeros((xfpred.shape[0], 2))
Zobs[:, 0] = xfobs                                      # Observation locations of f
Zpred[:, 0] = xfpred                                    # prediction locations of f

# Constructing the matrices
## Matrix K plus
n = xfobs.shape[0]
Kmat = lambda params: CovM(Zobs, **params)              # To create the K matrix with given kernel params.

##########################
## LOOCV optimization steps
##########################
ytrue = f(xfobs)
if hypropt == 1:
    bL = 1.0; bR = 1.5                                  # left and right bounds
    bounds = [bL, bR]                                   # To initialize theta (correlation parameter)
    n_iter = 1000                                       # Number of optimization iterations 
    loomsefn = lambda theta : LOOMSESK1D(theta, ytrue, Kmat, sigmaval = 1.0)
    thetaopt, optscore, history = adamopt(loomsefn, bounds, n_iter = n_iter)
    print(f'{thetaopt=:.2f}'); print(f'{optscore=:.4f}')
    plt.plot(range(n_iter), history); plt.show()        # To visualize LOOCV loss curve
    sigmaopt = LOOVARSK1D(ytrue, Kmat, dict(thetaval = thetaopt, sigmaval = 1.0))
    print(f'{sigmaopt=:.2f}')
else:
    print(f'Using the fixed kernel params (user inputs): {theta=:.2f}, {sigma=:.2f}')
    thetaopt = theta; sigmaopt = sigma

kernelkwargs = dict(thetaval = thetaopt, sigmaval = sigmaopt);
K = Kmat(kernelkwargs)                                  # LOOCV optimal K matrix

## Matrix H 
H = CovM(Zobs, Zpred, **kernelkwargs)                   # LOOCV optimal H matrix

# Final predictions
## Simple Kriging
nugget = 1e-10                                          # Nugget effect, to make the K matrix PSD 
I = np.eye(n); nI = nugget * I                          # Identity matrix; nugget * Identity
Kinv = np.linalg.solve(K + nI, I)                       # K inverse, `np.linalg.solve` generally preferred over `np.linalg.inverse`  
yhat = H.T @ Kinv @ ytrue                               # H^T K^{-1} Z

## MSE computation
MSE = np.mean((yhat - f(xfpred))**2)
print(f'MSE Simple Krig = {MSE.round(2)}')

## Uncertainty quantification: simple co-Kriging expression
BaseVar = 0.0                                           # To make sure Conditional variance estimate is positive. 
Kpred = CovM(Zpred, thetaval = thetaopt,                # Unconditional uncertainty 
                    sigmaval = sigmaopt)
CondVar = Kpred - (H.T @ Kinv @ H);                     # Minimal MSE 
CondSigma = np.sqrt(np.diag(CondVar) + BaseVar);

# Visualization
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12})                                                     
plt.figure(figsize = (12,10))
lkwargs = dict(ls = '-', lw = 1.5, alpha = 0.75, c = 'k')     
plt.plot(X, f(X), label = 'f(x) = sin (x)', **lkwargs);                     # True function
skwargs = dict(c = 'k', marker = 'X', s = 300)
plt.scatter(xfobs, f(xfobs), label = 'True observations', **skwargs);       # sparse TRUE observations (f)
plt.plot(X, X.shape[0]*[0], 'k--')                                          # An x-axis
lkwargs = dict(ls = '--', lw = 1.5, c = 'blue')                 
plt.plot(xfpred, yhat, label = 'Predicted', **lkwargs)                      # Predictions
cI = 2; low = yhat - (cI * CondSigma); high = yhat + (cI * CondSigma)       # Confidence interval factor
fkwargs = dict(color = 'b', alpha = 0.25);
plt.ylim(-1.15, 1.15)                                                       # ylim optional
plt.fill_between(xfpred, low, high, label = '$\pm 2\sigma$ CI', **fkwargs)            # Confidence intervals
plt.title(f'Simple Kriging 1D: {thetaopt=:.2f}, {sigmaopt=:.6f}')  
if label == 1: 
    plt.legend()
if figsave == 1:
    plt.savefig(IMGPATH+'SK1D.pdf', dpi = 300, bbox_inches = 'tight', pad_inches = 0)
plt.show()

# New figure to highlight the fact that prediction points are sparse. black circles with no fill inside (to indicate predictions). Uncertainty quantification as yerr.                                                     
plt.figure(figsize = (12,10))
lkwargs = dict(ls = '-', lw = 1.5, alpha = 0.75, c = 'k')     
plt.plot(X, f(X), label = 'f(x) = sin (x)', **lkwargs);                     # True function
skwargs = dict(c = 'k', marker = 'X', s = 300)
plt.scatter(xfobs, f(xfobs), label = 'True observations', **skwargs);       # sparse TRUE observations (f)
plt.plot(X, X.shape[0]*[0], 'k--')                                          # An x-axis
skwargs = dict(marker = 'o', s = 100, fc = 'none', ec = 'k', lw = 1.5)
plt.scatter(xfpred, yhat, label = 'Predicted', **skwargs)                   # Predictions

# # To show f + f'' = 0
# yddhat = np.gradient(np.gradient(yhat, xfpred), xfpred)                     # Predicted double derivative f''
# skwargs = dict(marker = 'o', s = 100, fc = 'none', ec = 'r', lw = 1.5)
# plt.scatter(xfpred, yhat + yddhat, label = 'Constraint', **skwargs)

cI = 2;                                                                     # Confidence interval factor
plt.ylim(-1.15, 1.15)                                                       # ylim optional
ekwargs = dict(color = 'b', fmt = 'none', alpha = 0.25, lw = 5)
plt.errorbar(xfpred, yhat, cI*CondSigma, label = '$\pm 2\sigma$ CI', **ekwargs)    # Confidence intervals
plt.title(f'Simple Kriging 1D: {thetaopt=:.2f}, {sigmaopt=:.6f}')  
if label == 1: 
    plt.legend()
if figsave == 1:
    plt.savefig(IMGPATH+'SK1D-discrete.pdf', dpi = 300, bbox_inches = 'tight', pad_inches = 0)
plt.show()