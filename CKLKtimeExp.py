# To check the run time of CK and LK for varying p (without LOOCV parameter tuning)
# Imports
import numpy as np
import time
from src.matrices import CovM, Umat, Kmat, Hmat

# Constants
PI = np.pi
seed = 20                                               # Reproducibility 
np.random.seed(seed)
printline = lambda *string: print(*string, '\n', 100 * '-' , '\n' , 100 * '-')  

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
sigma: float = 1.0                                      # Noise variance parameter
theta: float = 1.0                                      # Correlation parameter

# Creating important arrays
xfobs = np.array([0., 2.35330497,                       # Original code observations
                    4.59925358, 5.97351416])  

plist = [100, 250, 500, 1000]                           # p values to compute time for
# plist = [1, 2, 5, 10]                                   # p values to compute time for
Timelist = np.zeros((2, len(plist), 4))                 # to store the times: [Co-Krig OR LagKrig] X [p] X [K construction/ H construction/ K inv/ prediction] 

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

Iter = 0
for nfxx in plist:    
    printline(f'Iteration {Iter+1}/{len(plist)}')
    # Co-Kriging
    printline('Co-Kriging time')
    xfpred = np.linspace(*limits, 100)                  # Prediction locations of f (q = 100, fixed)
    xfxx = np.linspace(*limits, nfxx)                   # Randomly selected collocation points
    Zobs = np.zeros((xfobs.shape[0], 2))                # Extended design space: Z = (fval, m)
    Z = np.zeros((xfxx.shape[0], 2))
    Zxx = 2 * np.ones((xfxx.shape[0], 2))
    Zpred = np.zeros((xfpred.shape[0], 2))
    Zobs[:, 0] = xfobs                                  # Observation locations of f
    Z[:, 0] = xfxx                                      # f values at collocation locations
    Zxx[:, 0] = xfxx                                    # fxx values at collocation locations
    Zpred[:, 0] = xfpred                                # True f values at prediction locations 
    n = xfobs.shape[0]
    p = xfxx.shape[0]
    ExtZ = np.zeros((n + 2 * p, 2))                         # CAUTION: This might need changing. The factor of two appears because binary operation (f + f'')
    ExtZ[:n, :] = Zobs                             
    for _, i in zip(range(p), range(n, n + 2 * p, 2)):
        ExtZ[i, :] = Z[_, :]
        ExtZ[i + 1, :] = Zxx[_, :]

    ## Matrix U
    U = Umat([1,1], p)                                      # Coefficients 1 * f + 1 * f'' at p collocation locations
    Kmatfn = lambda params : Kmat(ExtZ, n, U, **params)     # Take kernel params [dict]

    ytrue = np.zeros(n + p)                                 # Dummy observations
    
    # MATRIX K^+ 
    CovStart = time.perf_counter()
    Kplus = Kmatfn(dict(thetaval = theta,                 
                         sigmaval = sigma))     
    CovEnd = time.perf_counter()
    print(f'p = {nfxx}: K^+ matrix construction time = {CovEnd - CovStart:.2f}s')
    Timelist[0, Iter, 0] = CovEnd - CovStart

    # Matrix H^+
    CovStart = time.perf_counter()
    Hplus = Hmat(ExtZ, n, Zpred, U, thetaval = theta,    # LOOCV optimal Hplus matrix 
                                    sigmaval = sigma) 
    CovEnd = time.perf_counter()
    print(f'p = {nfxx}, q = {xfpred.shape[0]}: H^+ matrix construction time = {CovEnd - CovStart:.2f}s')
    Timelist[0, Iter, 1] = CovEnd - CovStart

    # Final predictions
    ## Simple co-Kriging [yhat]
    nugget = 1e-10                                          # Nugget effect, to make the K matrix PSD 
    I = np.eye(n+p); nI = nugget * I                        # Identity matrix; nugget * Identity
    
    INVstart = time.perf_counter()
    Kplusinv = np.linalg.solve(Kplus + nI, I)               # K inverse, `np.linalg.solve` generally preferred over `np.linalg.inverse` 
    INVend = time.perf_counter()
    print(f'p = {nfxx}: K^+ matrix inversion time = {INVend - INVstart:.2f}s')
    Timelist[0, Iter, 2] = INVend - INVstart

    PRODstart = time.perf_counter()
    yhat = Hplus.T @ Kplusinv @ ytrue                       # (H^+)^T (K^+)^{-1} [Z, U^T Z^+]
    PRODend = time.perf_counter()
    print(f'p = {nfxx}: co-Krig prediction time = {PRODend - PRODstart:.2f}s')
    Timelist[0, Iter, 3] = PRODend - PRODstart
    printline()

    # Lagrangian Kriging
    printline('Lagrangian Kriging time')
    xfpred = np.linspace(*limits, nfxx)                     # Prediction locations of f (q = 100, fixed)
    Zobs = np.zeros((xfobs.shape[0], 2))                    # Extended design space: Z = (fval, m)
    Zpred = np.zeros((xfpred.shape[0], 2))                  # To predict: f  
    Zxxpred = 2 * np.ones((xfpred.shape[0], 2))             # To satisfy constraints involving: f''
    Zobs[:, 0] = xfobs
    Zpred[:, 0] = xfpred; Zxxpred[:, 0] = xfpred            # f, f'' at prediction locations 
    ExtZpred = np.zeros((2 * nfxx, 2))                      # CAUTION: This might need changing. The factor of two appears because binary operation (f + f'')
    for _, i in zip(range(nfxx), range(0, 2 * nfxx, 2)):
        ExtZpred[i, :] = Zpred[_, :]
        ExtZpred[i + 1, :] = Zxxpred[_, :]
    kernelkwargs = dict(thetaval = theta, sigmaval = sigma)
    ############################
    ## Constructing the matrices
    ############################
    ## Matrix K 
    n = xfobs.shape[0]
    CovStart = time.perf_counter()
    K = CovM(Zobs, **kernelkwargs)
    CovEnd = time.perf_counter()
    print(f'p = q = {nfxx}: K matrix construction time = {CovEnd - CovStart:.2f}s')
    Timelist[1, Iter, 0] = CovEnd - CovStart
    ## Matrix U
    q = xfpred.shape[0]                                     # of prediction locations
    U = Umat([1,1], q)                                      # Coefficients 1 * f + 1 * f'' at q prediction locations

    ## vector v
    v = np.zeros(q)                                         # RHS of constraint equations

    ## Matrix H
    CovStart = time.perf_counter()
    H = CovM(Zobs, ExtZpred, **kernelkwargs)                # Extended prediction covariance
    CovEnd = time.perf_counter()
    print(f'p = q = {nfxx}: H matrix construction time = {CovEnd - CovStart:.2f}s')
    Timelist[1, Iter, 1] = CovEnd - CovStart

    I = np.eye(n); nI = 1e-10 * I                           # nugget effect: nugget = 1e-10 to ensure PSD matrix
    INVstart = time.perf_counter()
    Kinv = np.linalg.solve(K + nI, I)                       # K matrix inverse
    INVend = time.perf_counter()
    print(f'p = q = {nfxx}: K matrix inversion time = {INVend - INVstart:.2f}s')
    Timelist[1, Iter, 2] = INVend - INVstart
    ##############################
    # Lagrangian Kriging Predictions
    ##############################
    ytrue = f(xfobs)
    PRODstart = time.perf_counter()
    AllPred = LagKrig(Kinv, H, v, ytrue, U)                 # all predcitions of f and f''.
    PRODend = time.perf_counter()
    print(f'p = q = {nfxx}: Lag Krig prediction time = {PRODend - PRODstart:.2f}s')
    Timelist[1, Iter, 3] = PRODend - PRODstart
    printline()
    Iter += 1

# print(f'Co Kriging timings: \n {Timelist[0,:,:].round(2)} \n')
# print(f'Co-Kriging total times = {np.sum(Timelist[0,:,:], axis = 1).round(2)}\n')
# print(f'Lagrangian Kriging timings: \n {Timelist[1,:,:].round(2)}\n')
# print(f'Lagrangian Kriging total times = {np.sum(Timelist[1,:,:], axis = 1).round(2)}')

# Total construction time and total inversion time
print(f'Co-Kriging construction times: \n {plist} \n {np.sum(Timelist[0,:,:2], axis = 1).round(4)} \n')
print(f'% times: \n p = {plist} \n {100 * np.sum(Timelist[0,:,:2], axis = 1)/np.sum(Timelist[0,:,:], axis = 1)} \n')
print(f'Co-Kriging inversion times: \n {plist} \n {np.sum(Timelist[0,:,2:], axis = 1).round(4)} \n')
print(f'% times: \n p = {plist} \n {100 * np.sum(Timelist[0,:,2:], axis = 1)/np.sum(Timelist[0,:,:], axis = 1)} \n')
print(f'Co-Kriging total time: \n {plist} \n {np.sum(Timelist[0,:,:], axis = 1).round(4)} \n')


print(f'Lagrangian Kriging construction times: \n {plist} \n {np.sum(Timelist[1,:,:2], axis = 1).round(4)} \n')
print(f'% times: \n q = {plist} \n {100 * np.sum(Timelist[1,:,:2], axis = 1)/np.sum(Timelist[1,:,:], axis = 1)} \n')
print(f'Lagrangian Kriging inversion times: \n {plist} \n {np.sum(Timelist[1,:,2:], axis = 1).round(4)} \n')
print(f'% times: \n q = {plist} \n {100 * np.sum(Timelist[1,:,2:], axis = 1)/np.sum(Timelist[1,:,:], axis = 1)} \n')
print(f'Lagrangian Kriging total time: \n {plist} \n {np.sum(Timelist[1,:,:], axis = 1).round(4)} \n')
