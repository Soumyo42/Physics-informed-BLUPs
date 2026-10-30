# Co Krging 2D scalar functions
# Imports
import numpy as np
from numpy import cos, sin, exp 
import matplotlib.pyplot as plt
from src.matrices import CovM, Umat
from src.covderivatives2D import AnalyticalSqExpdiff2D
from src.calibration import adamopt, LOOMSECK2D, LOOVARCK2D
import time 

start = time.time()                                         # To compute the full run time
###################

# Constants
PI = np.pi
seed = 1
np.random.seed(seed)                                        # Reproducibility

# Choose function(s)
f1flag = 1;                                                 # To launch code for f(x, y) = cos(x) sin(y)
f2flag = 0;                                                 # To launch code for f(x, y) = e^x sin(y) + 5    
UQflag = 0;                                                 # time consuming to compute the UQ

# Scalar Functions
f1 = lambda X, Y: cos(X) * sin(Y)                           # f(x, y) = cos(x) sin(y)
L1f1 = lambda X, Y: cos(X + Y)                              # fx(x, y) + fy(x, y) = cos(x) cos(y) - sin(x) sin(y) = cos(x + y)
L2f1 = lambda X, Y: -2 * f1(X, Y)                           # fxx(x, y) + fyy(x, y) =  - cos(x) sin(y) - cos(x) sin(y) = -2 * cos(x) sin (y)                                             

f2 = lambda X, Y: (exp(X) * sin(Y)) + 5                     # f(x, y) = e^x sin(y) + 5                  (Harmonic function)
L1f2 = lambda X, Y: exp(X) * (sin(Y) + cos(Y))              # fx(x, y) + fy(x, y) = e^x (sin(y) + cos(y))
L2f2 = lambda X, Y: f2(X, Y) - f2(X, Y)                     # fxx(x, y) + fyy(x, y) = 0

# Utility functions
gridpoints = lambda xl, xr, yd, yu, n: np.meshgrid(np.linspace(xl, xr, int((n)**(0.5))), 
                                                   np.linspace(yd, yu, int((n)**(0.5))))

# user inputs
xl = 0; xr = 2 * PI
yd = 0; yu = 2 * PI
nobs = 10
nL1 = 50; nL2 = 100                                         # min. number of collocation points 4
disc = 50; npred = 30
sigma = 1.0; theta = 1.41  
IMGPATH = r'IMAGES/'                                        # Image path

# Flag settings
gridobs = 0;                                                # Grid observations
gridp = 1;                                                  # Grid collocation points
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

xL1 = np.random.uniform(xl, xr, nL1)                        # Collocation locations for L1
yL1 = np.random.uniform(yd, yu, nL1)
if gridp == 1:
    xL1, yL1 = gridpoints(xl, xr, yd, yu, nL1)              # Grid collocation locations <= nL1
    nL1 = int(nL1**(0.5))**2
Zx = np.zeros((nL1, 2, 2)); Zy = np.zeros((nL1, 2, 2))                                    
Zx[:, 0, 0] = xL1.reshape(nL1); Zx[:, 0, 1] = yL1.reshape(nL1)  # Setting the locations 
Zy[:, 0, 0] = xL1.reshape(nL1); Zy[:, 0, 1] = yL1.reshape(nL1)  # Setting the locations
Zx[:, 1, 0] = 1; Zy[:, 1, 1] = 1;                           # Setting the right order of derivatives
xL2 = np.random.uniform(xl, xr, nL2)                        # Collocation locations for L2
yL2 = np.random.uniform(yd, yu, nL2)
if gridp == 1:
    xL2, yL2 = gridpoints(xl, xr, yd, yu, nL2)              # Grid collocation locations <= L2
    nL2 = int(nL2**(0.5))**2
Zxx = np.zeros((nL2, 2, 2)); Zyy = np.zeros((nL2, 2, 2))                                    
Zxx[:, 0, 0] = xL2.reshape(nL2); Zxx[:, 0, 1] = yL2.reshape(nL2)  # Setting the locations
Zyy[:, 0, 0] = xL2.reshape(nL2); Zyy[:, 0, 1] = yL2.reshape(nL2)  # Setting the locations
Zxx[:, 1, 0] = 2; Zyy[:, 1, 1] = 2                          # Setting the right order of derivatives

xpred = np.linspace(xl, xr, npred)                          # Prediction locations
ypred = np.linspace(xl, xr, npred)
xpred, ypred = np.meshgrid(xpred, ypred)
Zpred = np.zeros((npred**2, 2, 2))                          # Derivative order is (0,0)
Zpred[:,0,0] = xpred.reshape(npred**2); Zpred[:,0,1] = ypred.reshape(npred**2) # Setting the locations

ExtZ = np.zeros((nobs + 2 * nL1 + 2 * nL2, 2, 2))           # CAUTION: This might need changing. The factor of two appears because binary operations.                             
ExtZ[:nobs] = Zobs
for _, i in zip(range(nobs, nobs + 2 * nL1, 2), range(nL1)):
    ExtZ[_] = Zx[i]; ExtZ[_ + 1] = Zy[i]
for _, i in zip(range(nobs + 2 * nL1, nobs + 2 * nL1 + 2 * nL2, 2), range(nL2)):
    ExtZ[_] = Zxx[i]; ExtZ[_ + 1] = Zyy[i]

# Constructing the covariance matrices
## Matrix U
U1 = Umat([1,1], nL1); U2 = Umat([1,1], nL2)                # Differential coefficients associated with the two equations

## Kernel differentiation funtion
DiffKernel = lambda x, y, xindex, yindex, **kwargs: AnalyticalSqExpdiff2D(x, y, 
                                                                          xindex, yindex, 
                                                                          **kwargs)
############################################
## LOOCV optimization steps
############################################
v1, v2 = np.zeros((2, nL1 + nL2))                           # RHS of DiffEqns
v1[:nL1] = L1f1(xL1, yL1).flatten(); v1[nL1:] = L2f1(xL2, yL2).flatten();  
v2[:nL1] = L1f2(xL1, yL1).flatten(); v2[nL1:] = L2f2(xL2, yL2).flatten();
ytrue1, ytrue2 = np.zeros((2, nobs + nL1 + nL2))            # Exact function observations
ytrue1[:nobs] = f1(xfobs, yfobs).flatten(); ytrue1[nobs:] = v1; 
ytrue2[:nobs] = f2(xfobs, yfobs).flatten(); ytrue2[nobs:] = v2; 

## Matrix K^+
def Kmat2D(nobs, nL1, nL2, U1, U2, **params):
    n = nobs; p1 = nobs + 2 * nL1; p2 = nobs + 2 * nL1 + 2 * nL2;
    CovParam = lambda *Zvals: CovM(*Zvals, kernel = DiffKernel, **params)
    K11 = CovParam(ExtZ[:n]) 
    K12 = CovParam(ExtZ[:n], ExtZ[n:p1]) @ U1; K21 = K12.T
    K13 = CovParam(ExtZ[:n], ExtZ[p1:p2]) @ U2; K31 = K13.T
    K22 = U1.T @ CovParam(ExtZ[n:p1]) @ U1 
    K23 = U1.T @ CovParam(ExtZ[n:p1], ExtZ[p1:p2]) @ U2; K32 = K23.T
    K33 = U2.T @ CovParam(ExtZ[p1:p2]) @ U2
    Kplus = np.block([[K11, K12, K13],
                      [K21, K22, K23],
                      [K31, K32, K33]])
    return Kplus

Kmatfn = lambda params: Kmat2D(nobs, nL1, nL2, U1, U2, **params)

# MAIN CODE
if __name__ == "__main__":

    if f1flag == 1:
        if hypropt == 1:
            bL = 0.5; bR = 1.0                                  # left and right bounds
            bounds = [bL, bR]                                   # To initialize theta (correlation parameter)
            n_iter = 100
            ## f(x, y) = cos(x) sin(y)
            loomsefn1 = lambda theta: LOOMSECK2D(theta, ytrue1, nobs, Kmatfn, sigma = 1.0)
            thetaopt1, optscore1, history1 = adamopt(loomsefn1, bounds, n_iter = n_iter)
            print(f'{thetaopt1=:.2f}'); print(f'{optscore1=:.4f}')
            sigmaopt1 = LOOVARCK2D(ytrue1, nobs, Kmatfn, dict(theta = thetaopt1, sigma = 1.0))
            print(f'{sigmaopt1=:.2f}')
            plt.plot(range(n_iter), history1); plt.title(f'Loss curve with {thetaopt1=:.2f}, {sigmaopt1=:.2f}') 
            plt.savefig(IMGPATH+'loss2De1'); 
            # plt.show()      # To visualize LOOCV loss curve
        else: 
            print(f'Using the fixed kernel params (user inputs): {theta=:.2f}, {sigma=:.2f}')
            thetaopt1 = theta; sigmaopt1 = sigma
    
        ## Matrix Kplus
        ## f(x, y) = cos(x) sin(y)
        Kplus1 = Kmatfn(dict(theta = thetaopt1,                 # LOOCV Optimal Kplus matrix 
                             sigma = 1.0))        
    
        ## Matrix H^+
        q = Zpred.shape[0]                                      # Number of prediction locations
        n = nobs; p1 = nobs + 2 * nL1
        params1 = dict(theta = thetaopt1, sigma = 1.0)
        CovParam1 = lambda *Zvals: CovM(*Zvals, kernel = DiffKernel, **params1)
        H1 = CovParam1(ExtZ[:n], Zpred)
        H2 = U1.T @ CovParam1(ExtZ[n:p1], Zpred)
        H3 = U2.T @ CovParam1(ExtZ[p1:], Zpred)
        Hplus1 = np.block([[H1], [H2], [H3]])
    
        # Final predictions
        ## Simple co-Kriging [yhat]
        nugget = 1e-10
        I = np.eye(Kplus1.shape[0]); nI = nugget * I
        Kplusinv1 = np.linalg.solve(Kplus1 + nI, I)
        yhat1 = Hplus1.T @ Kplusinv1 @ ytrue1 
    
        if UQflag == 1:
            ## Uncertainty quantification: simple co-Kriging expression
            BaseVar = 1e-5                                          # To make sure Conditional variance estimate is positive. 
            Kpred = CovParam1(Zpred)                                # Unconditional uncertainty 
            CondVar1 = Kpred - (Hplus1.T @ Kplusinv1 @ Hplus1);     # Minimal MSE
            CondVar1 += BaseVar;                                    # Small quantity added to make it positive                         
            CondSigma1 = np.sqrt(sigmaopt1**2 * np.diag(CondVar1));
        
    if f2flag == 1:
        if hypropt == 1:
            bL = 0.5; bR = 1.0                                  # left and right bounds
            bounds = [bL, bR]                                   # To initialize theta (correlation parameter)
            n_iter = 100
            ## f(x, y) = e^x sin(y) + 5
            loomsefn2 = lambda theta: LOOMSECK2D(theta, ytrue2, nobs, Kmatfn, sigma = 1.0)
            thetaopt2, optscore2, history2 = adamopt(loomsefn2, bounds, n_iter = n_iter)
            print(f'{thetaopt2=:.2f}'); print(f'{optscore2=:.4f}')
            plt.plot(range(n_iter), history2); 
            sigmaopt2 = LOOVARCK2D(ytrue2, nobs, Kmatfn, dict(theta = thetaopt2, sigma = 1.0))
            print(f'{sigmaopt2=:.2f}')
            plt.figure()
            plt.savefig(IMGPATH+'loss2De2'); plt.title(f'Loss curve with {thetaopt2=:.2f}, {sigmaopt2=:.2f}')
            # plt.show()        # To visualize LOOCV loss curve
        else: 
            print(f'Using the fixed kernel params (user inputs): {theta=:.2f}, {sigma=:.2f}')
            thetaopt2 = theta; sigmaopt2 = sigma
    
        ## f(x, y) = e^x sin(y) + 5
        Kplus2 = Kmatfn(dict(theta = thetaopt2,                   # LOOCV Optimal Kplus matrix 
                            sigma = 1.0))
        
        ## Matrix H^+
        q = Zpred.shape[0]                                        # Number of prediction locations
        n = nobs; p1 = nobs + 2 * nL1
        params2 = dict(theta = thetaopt2, sigma = 1.0)
        CovParam2 = lambda *Zvals: CovM(*Zvals, kernel = DiffKernel, **params2)
        H1 = CovParam2(ExtZ[:n], Zpred)
        H2 = U1.T @ CovParam2(ExtZ[n:p1], Zpred)
        H3 = U2.T @ CovParam2(ExtZ[p1:], Zpred)
        Hplus2 = np.block([[H1], [H2], [H3]])
    
        # Final predictions
        ## Simple co-Kriging [yhat]
        nugget = 1e-10
        I = np.eye(Kplus2.shape[0]); nI = nugget * I
        Kplusinv2 = np.linalg.solve(Kplus2 + nI, I)
        yhat2 = Hplus2.T @ Kplusinv2 @ ytrue2 
    
        if UQflag == 1:
            ## Uncertainty quantification: simple co-Kriging expression
            BaseVar = 1e-5                                        # To make sure Conditional variance estimate is positive. 
            Kpred = CovParam2(Zpred)                              # Unconditional uncertainty 
            CondVar2 = Kpred - (Hplus2.T @ Kplusinv2 @ Hplus2);   # Minimal MSE
            CondVar2 = CondVar2 + BaseVar;                        # Small added variance to make it positive 
            CondSigma2 = np.sqrt(sigmaopt2**2 * np.diag(CondVar2));
        
    # Visualization
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12})       
    skwargs = dict(c = 'k', marker = 'X', s = 100)
    ckwargs1 = dict(facecolors='none', edgecolors='black', s = 100)
    ckwargs2 = dict(facecolors='none', edgecolors='red', s = 100)
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
        plt.colorbar()
        plt.scatter(xfobs, yfobs, label = 'Observations', **skwargs)
        plt.scatter(xL1, yL1, label = 'Collocation', **ckwargs1)
        plt.scatter(xL2, yL2, **ckwargs2)
        plt.title(f'Co-Kriging 2D: {thetaopt1=:.2f}, {sigmaopt1=:.6f}')  
        if label == 1: 
            plt.legend()
        if figsave == 1:
            plt.savefig(IMGPATH+'CK2De1.pdf', dpi = 300, bbox_inches = 'tight', pad_inches = 0)
        plt.show()
    
        if UQflag == 1:
            plt.figure(figsize = (12,10))
            plt.cla()
            plt.contourf(xpred, ypred, CondSigma1.reshape(npred, npred))
            plt.colorbar()
            plt.scatter(xfobs, yfobs, label = 'Observations', **skwargs)
            plt.title(f'Co-Kriging UQ 2D: {thetaopt1=:.2f}, {sigmaopt1=:.6f}') 
            if label == 1: 
                plt.legend()
            if figsave == 1:
                plt.savefig(IMGPATH+'CK2DUQe1.pdf', dpi = 300, bbox_inches = 'tight', pad_inches = 0)
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
        plt.colorbar()
        plt.scatter(xfobs, yfobs, label = 'Observations', **skwargs)
        plt.scatter(xL1, yL1, label = 'Collocation', **ckwargs1)
        plt.scatter(xL2, yL2, **ckwargs2)
        plt.title(f'Co-Kriging 2D: {thetaopt2=:.2f}, {sigmaopt2=:.6f}')
        if label == 1: 
            plt.legend()
        if figsave == 1:
            plt.savefig(IMGPATH+'CK2De2.pdf', dpi = 300, bbox_inches = 'tight', pad_inches = 0)
        plt.show()
        
        if UQflag == 1:
            plt.figure(figsize = (12,10))
            plt.cla()
            plt.contourf(xpred, ypred, CondSigma2.reshape(npred, npred))
            plt.colorbar()
            plt.scatter(xfobs, yfobs, label = 'Observations', **skwargs)
            plt.title(f'Co-Kriging 2D: {thetaopt2=:.2f}, {sigmaopt2=:.6f}')
            if label == 1: 
                plt.legend()
            if figsave == 1:
                plt.savefig(IMGPATH+'CK2DUQe2.pdf', dpi = 300, bbox_inches = 'tight', pad_inches = 0)
            plt.show()
    
    end = time.time()
    #################
    print(f'Program run time = {end - start:.2f}s')

nqL1 = 900; nqL2 = 900                                         # 30 X 30 grid to predict the derivatives (PredictingDerivatives.py)
if gridp == 1:
    xqL1, yqL1 = gridpoints(xl, xr, yd, yu, nqL1)              # Grid collocation locations <= nL1
    nqL1 = int(nqL1**(0.5))**2
    xqL2, yqL2 = gridpoints(xl, xr, yd, yu, nqL2)              # Grid collocation locations <= nL1
    nqL2 = int(nqL2**(0.5))**2

def PredDerivativesCK():
    thetaopt = theta;

    ## Matrix Kplus
    ## f(x, y) = cos(x) sin(y)
    Kplus = Kmatfn(dict(theta = thetaopt,                 # LOOCV Optimal Kplus matrix 
                         sigma = 1.0))
    
    Zx = np.zeros((nqL1, 2, 2)); Zy = np.zeros((nqL1, 2, 2))                                    
    Zx[:, 0, 0] = xqL1.reshape(nqL1); Zx[:, 0, 1] = yqL1.reshape(nqL1)  # Setting the locations 
    Zy[:, 0, 0] = xqL1.reshape(nqL1); Zy[:, 0, 1] = yqL1.reshape(nqL1)  # Setting the locations
    Zx[:, 1, 0] = 1; Zy[:, 1, 1] = 1;                           # Setting the right order of derivatives
    Zxx = np.zeros((nqL2, 2, 2)); Zyy = np.zeros((nqL2, 2, 2))                                    
    Zxx[:, 0, 0] = xqL2.reshape(nqL2); Zxx[:, 0, 1] = yqL2.reshape(nqL2)  # Setting the locations
    Zyy[:, 0, 0] = xqL2.reshape(nqL2); Zyy[:, 0, 1] = yqL2.reshape(nqL2)  # Setting the locations
    Zxx[:, 1, 0] = 2; Zyy[:, 1, 1] = 2                          # Setting the right order of derivatives

    ExtZpred = np.zeros((2 * nqL1 + 2 * nqL2, 2, 2))              # CAUTION: This might need changing. The factor of two appears because binary operations.                             
    for _, i in zip(range(0, 2 * nqL1, 2), range(nqL1)):
        ExtZpred[_] = Zx[i]; ExtZpred[_ + 1] = Zy[i]
    for _, i in zip(range(2 * nqL1, 2 * nqL1 + 2 * nqL2, 2), range(nqL2)):
        ExtZpred[_] = Zxx[i]; ExtZpred[_ + 1] = Zyy[i]
    
    ## Matrix H^+
    q = ExtZpred.shape[0]                                      # Number of prediction locations
    n = nobs; p1 = nobs + 2 * nL1
    params = dict(theta = thetaopt, sigma = 1.0)
    CovParam = lambda *Zvals: CovM(*Zvals, kernel = DiffKernel, **params)
    H1 = CovParam(ExtZ[:n], ExtZpred)
    H2 = U1.T @ CovParam(ExtZ[n:p1], ExtZpred)
    H3 = U2.T @ CovParam(ExtZ[p1:], ExtZpred)
    Hplus = np.block([[H1], [H2], [H3]])

    # Final predictions
    ## Simple co-Kriging [yhat]
    nugget = 1e-10
    I = np.eye(Kplus.shape[0]); nI = nugget * I
    Kplusinv = np.linalg.solve(Kplus + nI, I)
    Allpred = Hplus.T @ Kplusinv @ ytrue1
    
    ## Unpacking, rearranging the constrained predictions
    yxhat1, yyhat1 = np.zeros((2, nqL1));
    yxxhat1, yyyhat1 = np.zeros((2, nqL2));

    for _, i in zip(range(0, 2*nqL1, 2), range(nqL1)):
        if f1flag == 1:                    
            yxhat1[i] = Allpred[_];
            yyhat1[i] = Allpred[_ + 1]; 

    for _, i in zip(range(2*nqL1, 2*nqL1+2*nqL2,  2), range(nqL2)):
        if f1flag == 1:
            yxxhat1[i] = Allpred[_]; 
            yyyhat1[i] = Allpred[_ + 1]; 

    return yxhat1, yyhat1, yxxhat1, yyyhat1





    