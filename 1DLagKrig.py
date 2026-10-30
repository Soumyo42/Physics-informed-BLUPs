# 1D Lagrangian Kriging
# Imports
import numpy as np
import time
import matplotlib.pyplot as plt
from src.matrices import CovM, Umat, CovFn
from src.calibration import adamopt

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
nfpred: int = 1000                                      # prediction points, [= disc] for whole domain    
sigma: float = 3.91                                     # Noise variance parameter
theta: float =  2.38                                    # Correlation parameter
LOOCVMASK: bool = False                                 # False => LOOCV; True => Interpolation error minimization                      
IMGPATH = r'IMAGES/'                                    # Image path

# Flag settings
paperobs = 1;                                           # To use the same observation locations as the paper.
bdryobs = 1;                                            # 1 => To make the problem well posed, add one boundary observation, 0 => no boundary condition
hypropt = 0;                                            # 1 => LOOCV based covariance paramaters calibration, 0 => Fixed kernel params (to check results with a certain param set)
label = 0;                                              # 1 => To add a legend
figsave = 1;                                            # 1 => to save figs  
discrete = 0;                                           # Disrete point predictions.

# Creating important arrays
xfobs = np.random.uniform(*limits, nfobs)               # Randomly selected observation locations of f
if paperobs == 1:
    xfobs = np.array([0., 2.35330497,                   # Original code observations
                      4.59925358, 5.97351416])  
elif bdryobs == 1:                                      # Enables one boundary observation location
    xfobs = np.zeros(nfobs)
    xfobs[1:] = np.random.uniform(*limits, nfobs - 1)                
xfobs.sort(); xfpred = np.linspace(*limits, nfpred)     # Prediction locations of f
Zobs = np.zeros((xfobs.shape[0], 2))                    # Extended design space: Z = (loc, m)
Zpred = np.zeros((xfpred.shape[0], 2))                  # To predict: f  
Zxxpred = 2 * np.ones((xfpred.shape[0], 2))             # To satisfy constraints involving: f''    
Zobs[:, 0] = xfobs                                      # Observation locations of f
Zpred[:, 0] = xfpred; Zxxpred[:, 0] = xfpred            # f, f'' at prediction locations 
ExtZpred = np.zeros((2 * nfpred, 2))                    # CAUTION: This might need changing. The factor of two appears because binary operation (f + f'')
for _, i in zip(range(nfpred), range(0, 2 * nfpred, 2)):
    ExtZpred[i, :] = Zpred[_, :]
    ExtZpred[i + 1, :] = Zxxpred[_, :]

###########################
## Simple Lagrangian Kriging: final prediction
###########################
def LagKrig(Kinv, H, v, ytrue, U, **kwargs):
    '''
    LagKrig
    
    :param Kinv: K inverse
    :param H: H matrix
    :param v: RHS of constraints
    :param ytrue: True observations
    :param U: U matrix
    :param kwargs: Pass UTUinv

    Feel free to pass UTUinv as a keyword argument.
    '''
    if 'UTUinv' in kwargs:
        # print('Using U^T U inverse given')
        UTUinv = kwargs.get('UTUinv')
    else:
        # print('Computing U^T U inverse')
        UTU = U.T @ U
        I = np.eye(UTU.shape[0])
        UTUinv =  np.linalg.solve(UTU, I)
    HKZ = H.T @ Kinv @ ytrue                            # Simple Kriging prediction
    prediction = (U @ UTUinv @ (v - U.T @ HKZ)) + HKZ
    return prediction

################################################
## Simple Lagrangian Kriging: Uncertainty Quantification
################################################
def LagVar(Zpred, Kinv, H, v, ytrue, U, **kwargs):
    '''
    MMSE prediction (Conditional variance approximation)
    
    :param Zpred: Prediction locations and resp. orders of derivatives.
    :param Kinv: K inverse
    :param H: Matrix H
    :param v: RHS of constraints
    :param ytrue: Observed y values
    :param U: Coefficient matrix
    :param kwargs: Extra

    Feel free to pass `UTUinv`, `lamda`, `gamma`, `thetaopt` and `sigmaopt` as a keyword arguments. 
    '''
    theta = 1.0; sigma = 1.0                            # Default. Pass the optimal kernel params as keyword arguments
    if 'UTUinv' in kwargs:
        # print('Using U^T U inverse given')
        UTUinv = kwargs.get('UTUinv')
    else:
        # print('Computing U^T U inverse')
        UTU = U.T @ U
        I = np.eye(UTU.shape[0])
        UTUinv =  np.linalg.solve(UTU, I)               # (U^T U)^{-1} 
    if 'gamma' in kwargs:
        # print('Using gamma value given')
        gamma = kwargs.get('gamma')
    else:
        # print('Computing gamma value')
        gamma = ytrue.T @ Kinv @ ytrue                  # Z^T K^{-1} Z, RKHS norm
    HKZ = H.T @ Kinv @ ytrue                            # H^T K^{-1} Z, Simple Kriging prediction
    if 'lamda' in kwargs:
        # print('Using lambda (Lagrange multipliers) values given')
        lamda = kwargs.get('lamda')
    else:
        # print('Computing lambda values')
        lamda = (1/gamma) * UTUinv @ (v - (U.T @ HKZ))  # Lagrange multipliers
    
    ## Please pass optimal kernel parameters
    if 'thetaopt' in kwargs:
        theta = kwargs.get('thetaopt')
    if 'sigmaopt' in kwargs:
        sigma = kwargs.get('sigmaopt')

    ## final variance equation
    
    # The whole unconditional covariance matrix (q X q).  
    # Kstar = CovM(Zpred, thetaval = theta,               # Unconditional covariance.
    #                     sigmaval = sigma)

    # # Only the unconditional variance (diagonal of the covariance matrix)
    CovParam = lambda *Zvals: CovFn(*Zvals, thetaval = theta,   # Optimal coefficients
                                            sigmaval = sigma)   # Optimal coefficients
    Kstar = np.array(list(map(CovParam, Zpred, Zpred)))
    # print(f'{Kstar=}')
    # print(f'{np.diag(Kstar)=}')
    M1 = np.outer(ytrue, lamda) @ U.T
    M2 = (H + M1).T @ Kinv @ (M1 - H)                   # LK contribution
    # Var = np.diag(Kstar + M2)                         # IF whole unconditional covariance
    Var = Kstar + np.diag(M2)                           # IF only unconditional variance
    # print(f'{Var =}')
    BaseVar = 0.0                                       # To make sure Conditional variance estimate is positive. 
    return np.sqrt(Var + BaseVar)  

###########################
## LOOCV function
###########################
def LOOMSELK1D(theta, ytrue, Zobs, Covfn, LagKrig, nugget = 1e-10, **sigmakwargs):
    kernelkwargs = dict(thetaval = theta, **sigmakwargs)
    Z = Zobs.copy(); ExtPred = np.zeros((2, 2)) 
    MSELK = np.zeros(Z.shape[0]);    
    ## Matrix U
    U = Umat([1,1], 1); v = np.zeros(1)                 # Only one location; # RHS of the constraint equation
    UTU = U.T @ U; UTUinv = np.linalg.solve(UTU, np.eye(1))
    for _ in range(Z.shape[0]):
        mask = np.ones(len(Z), dtype = bool)
        mask[_] = LOOCVMASK; Obs = Z[mask]; 
        yobs = ytrue[mask]; 
        ExtPred[0] = Z[_];                              # Predicting Z at observation location i
        ExtPred[1] = Z[_] + np.array([0, 2]);           # Predicting Zxx at observation location i
        ## Matrix K
        K = Covfn(Obs, **kernelkwargs)                  # LOO K matrix
        ## Matrix H
        H = Covfn(Obs, ExtPred, **kernelkwargs)         # LOO H matrix
        N = K.shape[0]
        Kinv = np.linalg.solve(K + (np.eye(N) * nugget), np.eye(N))
        yhat = LagKrig(Kinv, H, v, yobs, U, UTUinv = UTUinv)[0]
        MSELK[_] = (ytrue[_] - yhat)**2
    return MSELK.mean()

###########################
###########################
def LOOVARLK1D(thetaOPT, ytrue, Zobs, Covfn, nugget = 1e-10):
    kernelkwargs = dict(thetaval = thetaOPT, sigmaval = 1.0)
    Z = Zobs.copy(); ExtPred = np.zeros((2, 2)) 
    WEIGHTEDMSELK = np.zeros(Z.shape[0])
    mask = np.ones(len(Z), dtype = bool)
    ## Matrix U
    U = Umat([1,1], 1); v = np.zeros(1)                 # Only one location; # RHS of the constraint equation
    UTU = U.T @ U; UTUinv = np.linalg.solve(UTU, np.eye(1))
    for _ in range(Z.shape[0]):
        mask[ _ ] = LOOCVMASK; Obs = Z[mask];           # LOOCV VAR criterion works best. 
        yobs = ytrue[mask];     
        ExtPred[0] = Z[_]; ExtPred[1] = Z[_] + [0, 2]   
        ## Matrix K
        K = Covfn(Obs, **kernelkwargs)                  # LOO K matrix
        ## Matrix H
        H = Covfn(Obs, ExtPred, **kernelkwargs)         # LOO H matrix
        N = K.shape[0]
        Kinv = np.linalg.solve(K + (np.eye(N) * nugget), np.eye(N))
        yhat = LagKrig(Kinv, H, v, yobs, U, UTUinv = UTUinv)[0] 
        ## Uncertainty Quantification (single point)
        gamma = yobs.T @ Kinv @ yobs + nugget
        lamda = (1/gamma) * UTUinv * (v - (U.T @ H.T @ Kinv @ yobs))
        M1 = np.outer(yobs, lamda) @ U.T
        M2 = (H + M1).T @ Kinv @ (M1 - H) 
        cii = np.diag(1 + M2)[0]
        WEIGHTEDMSELK[_] = ((ytrue[_] - yhat)**2)/cii
    sigma2opt = WEIGHTEDMSELK.mean()
    return np.sqrt(sigma2opt)

##########################
## LOOCV optimization steps
##########################
ytrue = f(xfobs)
if hypropt == 1:
    bL = 1.0; bR = 1.5                                  # left and right bounds
    bounds = [bL, bR]                                   # To initialize theta (correlation parameter)
    n_iter = 1000                                       # Number of optimization iterations 
    loomsefn = lambda theta : LOOMSELK1D(theta, ytrue, Zobs, CovM, LagKrig, sigmaval = 1.0)  
    thetaopt, optscore, history = adamopt(loomsefn, bounds, n_iter = n_iter)
    print(f'{thetaopt=:.2f}'); print(f'{optscore=:.4f}')
    plt.plot(range(n_iter), history); plt.show()        # To visualize LOOCV loss curve
    sigmaopt = LOOVARLK1D(thetaopt, ytrue, Zobs, CovM)  # LOOCV variance criterion
    print(f'{sigmaopt=:.2f}')
else:
    print(f'Using the fixed kernel params (user inputs): {theta=:.2f}, {sigma=:.2f}')
    thetaopt = theta; sigmaopt = sigma
kernelkwargs = dict(thetaval = thetaopt, sigmaval = sigmaopt)

############################
## Constructing the matrices
############################
## Matrix K 
n = xfobs.shape[0]
K = CovM(Zobs, **kernelkwargs)
I = np.eye(n); nI = 1e-10 * I                           # nugget effect: nugget = 1e-10 to ensure PSD matrix
Kinv = np.linalg.solve(K + nI, I)                       # K matrix inverse
    
## Matrix U
q = xfpred.shape[0]                                     # of prediction locations
U = Umat([1,1], q)                                      # Coefficients 1 * f + 1 * f'' at q prediction locations

## vector v
v = np.zeros(q)                                         # RHS of constraint equations

## Matrix H
H = CovM(Zobs, ExtZpred, **kernelkwargs)                # Extended prediction covariance

##############################
# Lagrangian Kriging Predictions
##############################
AllPred = LagKrig(Kinv, H, v, ytrue, U)                 # all predcitions of f and f''.
AllSigma = LagVar(ExtZpred, Kinv, H, v, ytrue, U,       # all standard devs of f and f''.
                  thetaopt = thetaopt, 
                  sigmaopt = sigmaopt)
yhat = np.zeros(nfpred); yxxhat = np.zeros(nfpred)      
ysig = np.zeros(nfpred); yxxsig = np.zeros(nfpred)
for _, i in zip(range(0, 2 * nfpred, 2), range(nfpred)):        
    yhat[i] = AllPred[_]; ysig[i] = AllSigma[_]         # rearranging the predictions of f and f'' separately. 
    yxxhat[i] = AllPred[_ + 1]; yxxsig[i] = AllSigma[_]

## MSE computation
MSE = np.mean((yhat - f(xfpred))**2)
print(f'MSE Lag Krig. = {MSE}')

###############################
# visualizing the predictions. 
###############################
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12})                                                     
plt.figure(figsize = (12,10))
lkwargs = dict(ls = '-', lw = 1.5, alpha = 0.75, c = 'k')                           
plt.plot(X, f(X), label = 'f(x) = sin (x)', **lkwargs);                     # True function
skwargs = dict(c = 'green', marker = 'o', s = 75)
# prediction locations in green
# plt.scatter(xfpred, nfpred * [0], label = 'Prediction locations', **skwargs);  # f + f'' = 0
skwargs = dict(c = 'k', marker = 'X', s = 300)
plt.scatter(xfobs, f(xfobs), label = 'True observations', **skwargs);       # sparse TRUE observations (f)
plt.plot(X, X.shape[0]*[0], 'k--')                                          # An x-axis

## f prediction
lkwargs = dict(ls = '--', lw = 1.5, c = 'blue')                 
plt.plot(xfpred, yhat, label = 'Predicted f', **lkwargs)                    #  Predictions
cI = 2; low = yhat - (cI * ysig); high = yhat + (cI * ysig)                 # Confidence interval factor
fkwargs = dict(color = 'b', alpha = 0.25);
plt.fill_between(xfpred, low, high, label = '$\pm 2\sigma$ CI', **fkwargs) # Confidence intervals

## f'' prediction
# lkwargs = dict(ls = '--', lw = 1.5, c = 'red')
# plt.plot(xfpred, yxxhat, label = 'Predicted f\'\'', **lkwargs)              
# cI = 2; low = yxxhat - (cI * yxxsig); high = yxxhat + (cI * yxxsig)       # Confidence interval factor
# fkwargs = dict(color = 'r', alpha = 0.25);
# plt.fill_between(xfpred, low, high, label = '$\pm 2\sigma CI$', **fkwargs)   # Confidence intervals

plt.ylim(-1.5, 1.5)                                                       # ylim optional
plt.title(f'Lagrangian Kriging 1D: {thetaopt=:.2f}, {sigmaopt=:.6f}')  
if label == 1: 
    plt.legend()
if figsave == 1:
    plt.savefig(IMGPATH+'LK1D.pdf', dpi = 300, bbox_inches = 'tight', pad_inches = 0)
plt.show()


# New figure to highlight the fact that prediction points are sparse. black circles with no fill inside (to indicate predictions). Uncertainty quantification as yerr.                                                     
if discrete == 1:
    plt.figure(figsize = (12,10))
    lkwargs = dict(ls = '-', lw = 1.5, alpha = 0.75, c = 'k')     
    plt.plot(X, f(X), label = 'f(x) = sin (x)', **lkwargs);                     # True function
    skwargs = dict(c = 'k', marker = 'X', s = 300)
    plt.scatter(xfobs, f(xfobs), label = 'True observations', **skwargs);       # sparse TRUE observations (f)
    plt.plot(X, X.shape[0]*[0], 'k--')                                          # An x-axis
    skwargs = dict(marker = 'o', s = 100, fc = 'none', ec = 'k', lw = 1.5)
    # f predictions
    plt.scatter(xfpred, yhat, label = 'Predicted f', **skwargs)                 #  Predictions
    cI = 2;                                                                     # Confidence interval factor
    plt.ylim(-1.15, 1.15)                                                       # ylim optional
    ekwargs = dict(color = 'b', fmt = 'none', alpha = 0.25, lw = 5)
    plt.errorbar(xfpred, yhat, cI*ysig, label = '$\pm 2\sigma$ CI', **ekwargs)  # Confidence intervals
    plt.title(f'Lagrangian Kriging 1D: {thetaopt=:.2f}, {sigmaopt=:.6f}')  
    if label == 1: 
        plt.legend()
    if figsave == 1:
        plt.savefig(IMGPATH+'LK1D-discrete.pdf', dpi = 300, bbox_inches = 'tight', pad_inches = 0)
    plt.show()
    


# if __name__ == '__main__':
#     # Testing: Constructed K matrix symmetric?
#     K = CovM(Zobs, sigmaval = sigma, thetaval = theta)
#     print(f'K matrix shape (n = {nfobs}): {K.shape}')
#     print(f'Is the K matrix symmetric?: {(K == K.T).all()}') 
#     # Testing: Is the U matrix correct?
#     print(f'For f+f\'\' = 0 and q = {q}, U= \n {U}')
#     print(f'shape of U: {U.shape}')
#     # Testing: shape of the H matrix
#     print(f'H shape (n = {n}), (q = {q}) = {H.shape}')
#     # Testing: The 'LagKrig' function
#     K = CovM(Zobs, sigmaval = sigma, thetaval = theta)
#     Kinv = np.linalg.inv(K + 1e-10)
#     U = Umat([1,1], nfpred)
#     UTU = U.T @ U; UTUinv = np.linalg.solve(UTU, np.eye(nfpred))
#     H = CovM(Zobs, ExtZpred, sigmaval = sigma, thetaval = theta)
#     v = np.zeros(nfpred)
#     predictions = LagKrig(Kinv, H, v, f(xfobs), U)          # Add `UTUinv` and see the difference
#     print(f'Prediction shape (q = {nfpred}) = {predictions.shape}')
#     print(f'Final predictions: {predictions}')
#     yhat = np.zeros(nfpred); yxxhat = np.zeros(nfpred)
#     for _, i in zip(range(0, 2 * nfpred, 2), range(nfpred)):
#         yhat[i] = predictions[_]
#         yxxhat[i] = predictions[_ + 1]
#     plt.plot(xfpred, yhat)
#     plt.show()
#     # Testing: LOOCV MSE
#     U = Umat([1,1], nfpred)
#     print(LOOMSELK1D(1.0, f(xfobs), Zobs, CovM, U, LagKrig, sigmaval = 1.0))