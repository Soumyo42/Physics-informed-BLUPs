# 1D Co-Kriging
# Imports
import numpy as np
import time
import matplotlib.pyplot as plt
from src.matrices import CovM, Umat, Kmat, Hmat
from src.calibration import adamopt, LOOMSECK1D, LOOVARCK1D

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
nfxx: int = 10                                          # Observations of the second derivatives
nfpred: int = 10                                        # prediction points, [= disc] for whole domain    
sigma: float = 0.006                                    # Noise variance parameter
theta: float = 1.46                                     # Correlation parameter
IMGPATH = r'IMAGES/'                                    # Image path

# Flag settings
paperobs = 1;                                           # To use the same observation locations as the paper.
gridp = 1;                                              # 1 => equispaced collocation points, 0 => uniform random collocation points
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

xfxx = np.random.uniform(*limits, nfxx)                 # Randomly selected collocation points
if gridp == 1:                                          # Enables grid collocation points
    xfxx = np.linspace(*limits, nfxx)                   
xfobs.sort(); xfxx.sort()
xfpred = np.linspace(*limits, nfpred)                   # Prediction locations of f

Zobs = np.zeros((xfobs.shape[0], 2))                    # Extended design space: Z = (fval, m)
Z = np.zeros((xfxx.shape[0], 2))
Zxx = 2 * np.ones((xfxx.shape[0], 2))
Zpred = np.zeros((xfpred.shape[0], 2))
Zobs[:, 0] = xfobs                                      # Observation locations of f
Z[:, 0] = xfxx                                          # f values at collocation locations
Zxx[:, 0] = xfxx                                        # fxx values at collocation locations
Zpred[:, 0] = xfpred                                    # True f values at prediction locations 

# Constructing the matrices
## Matrix K plus
n = xfobs.shape[0]
p = xfxx.shape[0]
ExtZ = np.zeros((n + 2 * p, 2))                         # CAUTION: This might need changing. The factor of two appears because binary operation (f + f'')
ExtZ[:n, :] = Zobs                             
for _, i in zip(range(p), range(n, n + 2 * p, 2)):
    ExtZ[i, :] = Z[_, :]
    ExtZ[i + 1, :] = Zxx[_, :]

## Matrix U
U = Umat([1,1], p)                                      # Coefficients 1 * f + 1 * f'' at p collocation locations
Kmatfn = lambda params : Kmat(ExtZ, n, U, **params)     # Take kernel params [dict] => K matrix

#################################################################
## LOOCV optimization steps
#################################################################
v = np.zeros(p)                                         # RHS of the DiffEqns
ytrue = np.zeros(n + p)
ytrue[:n], ytrue[n:] = f(xfobs), v                      # True values          

if hypropt == 1:
    bL = 0.5; bR = 1.0                                  # left and right bounds
    bounds = [bL, bR]                                   # To initialize theta (correlation parameter)
    n_iter = 50
    loomsefn = lambda theta : LOOMSECK1D(theta, ytrue, n, Kmatfn,  sigmaval = 1.0) 
    thetaopt, optscore, history = adamopt(loomsefn, bounds, n_iter = n_iter)
    print(f'{thetaopt=:.2f}'); print(f'{optscore=:.4f}')
    plt.plot(range(n_iter), history); plt.show()        # To visualize LOOCV loss curve
    sigmaopt = LOOVARCK1D(ytrue, n, Kmatfn, dict(thetaval = thetaopt, sigmaval = 1.0))
    print(f'{sigmaopt=:.2f}')
else: 
    print(f'Using the fixed kernel params (user inputs): {theta=:.2f}, {sigma=:.2f}')
    thetaopt = theta; sigmaopt = sigma

Kplus = Kmatfn(dict(thetaval = thetaopt,                # LOOCV Optimal Kplus matrix 
                     sigmaval = 1.0))              

## Matrix H
Hplus = Hmat(ExtZ, n, Zpred, U, thetaval = thetaopt,    # LOOCV optimal Hplus matrix 
                                sigmaval = 1.0) 

# Final predictions
## Simple co-Kriging [yhat]
nugget = 1e-10                                          # Nugget effect, to make the K matrix PSD 
I = np.eye(n+p); nI = nugget * I                        # Identity matrix; nugget * Identity
Kplusinv = np.linalg.solve(Kplus + nI, I)               # K inverse, `np.linalg.solve` generally preferred over `np.linalg.inverse`  
yhat = Hplus.T @ Kplusinv @ ytrue                       # (H^+)^T (K^+)^{-1} [Z, U^T Z^+]

## MSE computation
MSE = np.mean((yhat - f(xfpred))**2)
print(f'MSE Co-Krig = {MSE}')

## Uncertainty quantification: simple co-Kriging expression
BaseVar = 2*1e-4                                        # To make sure Conditional variance estimate is positive. 
Kpred = CovM(Zpred, thetaval = thetaopt,                # Unconditional uncertainty 
                    sigmaval = 1.0)
CondVar = sigmaopt**2*(Kpred - (Hplus.T @ Kplusinv @ Hplus));    # Minimal MSE
# print(f'uncertainty without sigmaopt: {np.diag(CondVar)}') 
CondSigma = np.sqrt(np.diag(CondVar) + BaseVar);

# Visualization
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12})                                                     
plt.figure(figsize = (12,10))
lkwargs = dict(ls = '-', lw = 1.5, alpha = 0.75, c = 'k')                           
plt.plot(X, f(X), label = 'f(x) = sin (x)', **lkwargs);                     # True function
skwargs = dict(c = 'green', marker = 'o', s = 75)
plt.scatter(xfxx, nfxx * [0], label = 'Collocation locations', **skwargs);  # f + f'' = 0
skwargs = dict(c = 'k', marker = 'X', s = 300)
plt.scatter(xfobs, f(xfobs), label = 'True observations', **skwargs);       # sparse TRUE observations (f)
plt.plot(X, X.shape[0]*[0], 'k--')                                          # An x-axis
lkwargs = dict(ls = '--', lw = 1.5, c = 'blue')                 
plt.plot(xfpred, yhat, label = 'Predicted', **lkwargs)                      # Predictions
cI = 2; low = yhat - (cI * CondSigma); high = yhat + (cI * CondSigma)       # Confidence interval factor
fkwargs = dict(color = 'b', alpha = 0.25);
plt.ylim(-1.15, 1.15)                                                       # ylim optional
plt.fill_between(xfpred, low, high, label = '$\pm 2\sigma$ CI', **fkwargs)  # Confidence intervals
plt.title(f'Co-Kriging 1D: {thetaopt=:.2f}, {sigmaopt=:.6f}')  
if label == 1: 
    plt.legend()
if figsave == 1:
    plt.savefig(IMGPATH+'CK1D.pdf', dpi = 300, bbox_inches = 'tight', pad_inches = 0)
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
plt.title(f'Co Kriging 1D: {thetaopt=:.2f}, {sigmaopt=:.6f}')  
if label == 1: 
    plt.legend()
if figsave == 1:
    plt.savefig(IMGPATH+'CK1D-discrete.pdf', dpi = 300, bbox_inches = 'tight', pad_inches = 0)
plt.show()







# if __name__ == '__main__':
#     # Testing: Matrix symmetric or not
#     K = CovM(ExtZ)
#     print(f'Is the constructed covariance matrix symmetric?: {(K == K.T).all()}')
#     # Testing: Matrix with two inputs
#     H = CovM(ExtZ, Zpred)
#     print(f'Shape with two inputs (n = {nfobs + 2 * nfxx}), (q = {xfpred.shape[0]}): {H.shape}')
#     # Testing: Kplus matrix
#     Params = dict(thetaval = theta, sigmaval = sigma)
#     CovParam = lambda *Zvals: CovM(*Zvals, **Params)  
#     K11 = CovParam(ExtZ[:n])                                # Simple Block structure 
#     K12 = CovParam(ExtZ[:n], ExtZ[n:]) @ U
#     K21 = K12.T
#     K22 = U.T @ CovParam(ExtZ[n:]) @ U
#     Kplus = np.block([[K11, K12],                           # [Cov(Z,Z) Cov(Z, Z^+)]
#                     [K21, K22]])                            # [Cov(Z^+, Z) Cov(Z^+, Z^+)]
#     print(f'Dimensions of the final Kplus matrix (n = {n}) (p = {p}): {Kplus.shape}')
    