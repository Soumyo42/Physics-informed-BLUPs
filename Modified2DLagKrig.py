# Lagrangian Krging 2D scalar functions
# Imports
import numpy as np
from numpy import cos, sin, exp 
import matplotlib.pyplot as plt
from src.matrices import CovM, Umat
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
f1flag = 1;                                                 # To launch code for f(x, y) = cos(x) sin(y)
f2flag = 0;                                                 # To launch code for f(x, y) = e^x sin(y) + 5    
UQflag = 0;                                                 # time consuming to compute the UQ

# Scalar Functions
f1 = lambda X, Y: cos(X) * sin(Y)                           # f(x, y) = cos(x) sin(y)
L1f1 = lambda X, Y: cos(X + Y)                              # fx(x, y) + fy(x, y) = cos(x) cos(y) - sin(x) sin(y) = cos(x + y)
L2f1 = lambda X, Y: -2 * f1(X, Y)                           # fxx(x, y) + fyy(x, y) =  - cos(x) sin(y) - cos(x) sin(y) = -2 * cos(x) sin (y)                                             

# f1 = lambda X, Y: (exp(X) * sin(Y)) + 5                     # f(x, y) = e^x sin(y) + 5                  (Harmonic function)
# L1f1 = lambda X, Y: exp(X) * (sin(Y) + cos(Y))              # fx(x, y) + fy(x, y) = e^x (sin(y) + cos(y))
# L2f1 = lambda X, Y: f1(X, Y) - f1(X, Y)                     # fxx(x, y) + fyy(x, y) = 0

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
nL1 = 50; nL2 = 50                                          # min. number of collocation points 4
disc = 50; 
sigma = 1.0; theta = 1.0  
IMGPATH = r'IMAGES/'                                        # Image path

# Flag settings
gridobs = 0;                                                # Grid observations
gridp = 1;                                                  # Grid collocation points
hypropt = 0;                                                # 1 => LOOCV based covariance paramaters calibration, 0 => Fixed kernel params (to check results with a certain param set)
label = 1;                                                  # 1 => To add a legend
figsave = 0;                                                # 1 => to save figs  

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

ExtZpred = np.zeros((2 * nL1 + 2 * nL2, 2, 2))              # CAUTION: This might need changing. The factor of two appears because binary operations.                             
for _, i in zip(range(0, 2 * nL1, 2), range(nL1)):
    ExtZpred[_] = Zx[i]; ExtZpred[_ + 1] = Zy[i]
for _, i in zip(range(2 * nL1, 2 * nL1 + 2 * nL2, 2), range(nL2)):
    ExtZpred[_] = Zxx[i]; ExtZpred[_ + 1] = Zyy[i]

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

###########################
## Simple Kriging: final prediction
###########################
def SimKrig(Kinv, H, ytrue, **kwargs):
    '''
    SimpleKrig
    
    :param Kinv: K inverse
    :param H: H matrix
    :param ytrue: True observations

    '''
    HKZ = H.T @ Kinv @ ytrue                            # Simple Kriging prediction
    prediction = HKZ
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
    Kstar = CovM(Zpred, kernel = DiffKernel, 
                        theta = theta,                  # Unconditional variance.
                        sigma = sigma) 
    M1 = np.outer(ytrue, lamda) @ U.T
    M2 = (H + M1).T @ Kinv @ (M1 - H)                   # LK contribution
    Var = np.diag(Kstar + M2)
    BaseVar = 1e-5                                      # To make sure Conditional variance estimate is positive. 
    return np.sqrt(Var + BaseVar)  
##############################################
# Constructing the covariance matrices
## Matrix U
U = Umat([1,1], nL1 + nL2) 

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
ytrue1, ytrue2 = np.zeros((2, nobs))                        # Exact function observations
ytrue1 = f1(xfobs, yfobs).flatten();
ytrue2 = f2(xfobs, yfobs).flatten();

## true partial derivatives. 
trueyx1 = -sin(xL1) * sin(yL1); trueyy1 = cos(xL1) * cos(yL1);
trueyxx1 = -f1(xL2, yL2); trueyyy1 = -f1(xL2, yL2);
trueyx2 = exp(xL1)*sin(yL1); trueyy2 = exp(xL1)*cos(yL1);
trueyxx2 = exp(xL2)*sin(yL2); trueyyy2 = -exp(xL2)*sin(yL2);

## Matrix K
def Kmat(params):
    CovParam = lambda *Zvals: CovM(*Zvals, kernel = DiffKernel, **params)
    K = CovParam(Zobs) 
    return K



if __name__ == "__main__":

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
        K = Kmat(dict(theta = thetaopt1,                         # LOOCV Optimal Kplus matrix 
                        sigma = sigmaopt1))        
    
        ## Matrix H
        q = ExtZpred.shape[0]                                    # Number of prediction locations
        params1 = dict(theta = thetaopt1, sigma = sigmaopt1)
        CovParam1 = lambda *Zvals: CovM(*Zvals, kernel = DiffKernel, **params1)
        H1 = CovParam1(Zobs, ExtZpred)
        
        # Final predictions
        ## Lagrangian Kriging [yhat]
        nugget = 1e-10
        I = np.eye(K.shape[0]); nI = nugget * I
        Kinv1 = np.linalg.solve(K + nI, I)
        Allpred1 = LagKrig(Kinv1, H1, v1, ytrue1, U) 
        
        if UQflag == 1:
            ## Uncertainty quantification: simple Lagrangian Kriging expression
            AllCondSigma1 = LagVar(ExtZpred, Kinv1, H1, v1, ytrue1, U)
    
    if f2flag == 1:
        if hypropt == 1:
                bL = 1.0; bR = 1.5                                  # left and right bounds
                bounds = [bL, bR]                                   # To initialize theta (correlation parameter)
                n_iter = 1000
                ## f(x, y) = cos(x) sin(y)
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
        ## f(x, y) = cos(x) sin(y)
        K = Kmat(dict(theta = thetaopt2,                         # LOOCV Optimal Kplus matrix 
                        sigma = sigmaopt2))        
    
        ## Matrix H
        q = ExtZpred.shape[0]                                    # Number of prediction locations
        params2 = dict(theta = thetaopt2, sigma = sigmaopt2)
        CovParam2 = lambda *Zvals: CovM(*Zvals, kernel = DiffKernel, **params2)
        H2 = CovParam2(Zobs, ExtZpred)
        
        # Final predictions
        ## Lagrangian Kriging [yhat]
        nugget = 1e-10
        I = np.eye(K.shape[0]); nI = nugget * I
        Kinv2 = np.linalg.solve(K + nI, I)
        Allpred2 = LagKrig(Kinv2, H2, v2, ytrue2, U) 
        
        if UQflag == 1:
            ## Uncertainty quantification: simple Lagrangian Kriging expression
            AllCondSigma2 = LagVar(ExtZpred, Kinv2, H2, v2, ytrue2, U)
    
    # Visualization
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12})       
    skwargs = dict(c = 'k', marker = 'X', s = 100)
    
    ## Unpacking, rearranging the constrained predictions
    yxhat1, yyhat1, yxhat2, yyhat2 = np.zeros((4, nL1));
    yxsig1, yysig1, yxsig2, yysig2 = np.zeros((4, nL1));
    yxxhat1, yyyhat1, yxxhat2, yyyhat2 = np.zeros((4, nL2));
    yxxsig1, yyysig1, yxxsig2, yyysig2 = np.zeros((4, nL2));
    
    for _, i in zip(range(0, 2*nL1, 2), range(nL1)):
        if f1flag == 1:                    
            yxhat1[i] = Allpred1[_];
            if UQflag == 1: 
                yxsig1[i] = AllCondSigma1[_];
            yyhat1[i] = Allpred1[_ + 1]; 
            if UQflag == 1:
                yysig1[i] = AllCondSigma1[_ + 1];
        if f2flag == 1:
            yxhat2[i] = Allpred2[_]; 
            if UQflag == 1:
                yxsig2[i] = AllCondSigma2[_];
            yyhat2[i] = Allpred2[_ + 1]; 
            if UQflag == 1:
                yysig2[i] = AllCondSigma2[_ + 1];
    
    for _, i in zip(range(2*nL1, 2*nL1+2*nL2,  2), range(nL2)):
        if f1flag == 1:
            yxxhat1[i] = Allpred1[_]; 
            if UQflag == 1:
                yxxsig1[i] = AllCondSigma1[_];
            yyyhat1[i] = Allpred1[_ + 1]; 
            if UQflag == 1:
                yyysig1[i] = AllCondSigma1[_ + 1];
        if f2flag == 1:
            yxxhat2[i] = Allpred2[_]; 
            if UQflag == 1:
                yxxsig2[i] = AllCondSigma2[_];
            yyyhat2[i] = Allpred2[_ + 1]; 
            if UQflag == 1:
                yyysig2[i] = AllCondSigma2[_ + 1];
    
    # Plotting f1(x, y) = cos(x) sin(y)
    if f1flag == 1:
        fig, ((ax1, ax2), 
              (ax3, ax4),
              (ax5, ax6), 
              (ax7, ax8)) = plt.subplots(4, 2, figsize = (12,10), sharex = 'col')
        plt.cla()
        plt.suptitle(f'Lagrangian Kriging 2D: {thetaopt1=:.2f}, {sigmaopt1=:.6f}')
        c1 = ax1.contourf(xL1, yL1, trueyx1.reshape(int((nL1)**(0.5)), int((nL1)**(0.5))))
        c2 = ax2.contourf(xL1, yL1, yxhat1.reshape(int((nL1)**(0.5)), int((nL1)**(0.5))))
        c3 = ax3.contourf(xL1, yL1, trueyy1.reshape(int((nL1)**(0.5)), int((nL1)**(0.5))))
        c4 = ax4.contourf(xL1, yL1, yyhat1.reshape(int((nL1)**(0.5)), int((nL1)**(0.5))))
        c5 = ax5.contourf(xL2, yL2, trueyxx1.reshape(int((nL2)**(0.5)), int((nL2)**(0.5))))
        c6 = ax6.contourf(xL2, yL2, yxxhat1.reshape(int((nL2)**(0.5)), int((nL2)**(0.5))))
        c7 = ax7.contourf(xL2, yL2, trueyyy1.reshape(int((nL2)**(0.5)), int((nL2)**(0.5))))
        c8 = ax8.contourf(xL2, yL2, yyyhat1.reshape(int((nL2)**(0.5)), int((nL2)**(0.5))))
        # axes = ((ax1, ax2), 
        #         (ax3, ax4),
        #         (ax5, ax6), 
        #         (ax7, ax8))
        for ax in fig.get_axes():
            ax.label_outer()
        for c in [c1, c2, c3, c4, c5, c6, c7, c8]:
            c.set_clim(-1.0, 1.0)
        plt.colorbar(c1, ax = ax1)
        plt.colorbar(c2, ax = ax2)
        plt.colorbar(c3, ax = ax3)
        plt.colorbar(c4, ax = ax4)
        plt.colorbar(c5, ax = ax5)
        plt.colorbar(c6, ax = ax6)
        plt.colorbar(c7, ax = ax7)
        plt.colorbar(c8, ax = ax8)
        # fig.colorbar(c1, ax=axes.ravel().tolist())
        ax1.set_title('Actual $\partial_xf_1$')
        ax2.set_title('Predicted $\partial_xf_1$')
        ax3.set_title('Actual $\partial_yf_1$')
        ax4.set_title('Predicted $\partial_yf_1$')
        ax5.set_title('Actual $\partial^2_{xx}f_1$')
        ax6.set_title('Predicted $\partial^2_{xx}f_1$')
        ax7.set_title('Actual $\partial^2_{yy}f_1$')
        ax8.set_title('Predicted $\partial^2_{yy}f_1$')
        if figsave == 1:
            plt.savefig(IMGPATH+'LK2De1.pdf', dpi = 300, bbox_inches = 'tight', pad_inches = 0)
        plt.show()
    
    # Plotting f2(x, y) = e^x sin(y) + 5
    if f2flag == 1:
        fig, ((ax1, ax2), 
              (ax3, ax4),
              (ax5, ax6), 
              (ax7, ax8)) = plt.subplots(4, 2, figsize = (12,10), sharex = 'col')
        plt.cla()
        plt.suptitle(f'Lagrangian Kriging 2D: {thetaopt2=:.2f}, {sigmaopt2=:.6f}')
        c1 = ax1.contourf(xL1, yL1, trueyx2.reshape(int((nL1)**(0.5)), int((nL1)**(0.5))))
        c2 = ax2.contourf(xL1, yL1, yxhat2.reshape(int((nL1)**(0.5)), int((nL1)**(0.5))))
        c3 = ax3.contourf(xL1, yL1, trueyy2.reshape(int((nL1)**(0.5)), int((nL1)**(0.5))))
        c4 = ax4.contourf(xL1, yL1, yyhat2.reshape(int((nL1)**(0.5)), int((nL1)**(0.5))))
        c5 = ax5.contourf(xL2, yL2, trueyxx2.reshape(int((nL2)**(0.5)), int((nL2)**(0.5))))
        c6 = ax6.contourf(xL2, yL2, yxxhat2.reshape(int((nL2)**(0.5)), int((nL2)**(0.5))))
        c7 = ax7.contourf(xL2, yL2, trueyyy2.reshape(int((nL2)**(0.5)), int((nL2)**(0.5))))
        c8 = ax8.contourf(xL2, yL2, yyyhat2.reshape(int((nL2)**(0.5)), int((nL2)**(0.5))))
        # axes = ((ax1, ax2), 
        #         (ax3, ax4),
        #         (ax5, ax6), 
        #         (ax7, ax8))
        for ax in fig.get_axes():
              ax.label_outer()
        for c in [c1, c2, c3, c4, c5, c6, c7, c8]:
            c.set_clim(-600, 600)
        plt.colorbar(c1, ax = ax1)
        plt.colorbar(c2, ax = ax2)
        plt.colorbar(c3, ax = ax3)
        plt.colorbar(c4, ax = ax4)
        plt.colorbar(c5, ax = ax5)
        plt.colorbar(c6, ax = ax6)
        plt.colorbar(c7, ax = ax7)
        plt.colorbar(c8, ax = ax8)
        # fig.colorbar(c1, ax=axes.ravel().tolist())
        ax1.set_title('Actual $\partial_xf_2$')
        ax2.set_title('Predicted $\partial_xf_2$')
        ax3.set_title('Actual $\partial_yf_2$')
        ax4.set_title('Predicted $\partial_yf_2$')
        ax5.set_title('Actual $\partial^2_{xx}f_2$')
        ax6.set_title('Predicted $\partial^2_{xx}f_2$')
        ax7.set_title('Actual $\partial^2_{yy}f_2$')
        ax8.set_title('Predicted $\partial^2_{yy}f_2$')
        if figsave == 1:
            plt.savefig(IMGPATH+'LK2De2.pdf', dpi = 300, bbox_inches = 'tight', pad_inches = 0)
        plt.show()

## TODO Uncertainty quantification figures. 

def PredDerivatives(mode = 0):
    '''
    Predicting derivatives with LK

    :param mode: int = 0 OR 1 (0 returns predictions in a sorted manner, 1 returns criss-cross (like ExtZpred) predictions.)
    :return: Derivative predictions 
    '''
    thetaopt = theta;
    ## Matrix K
    print(f'Using the fixed kernel params (user inputs): {thetaopt=:.2f}')
    K = Kmat(dict(theta = thetaopt,                         # Kplus matrix 
                    sigma = 1.0))   
    
    ## Matrix H
    q = ExtZpred.shape[0]                                    # Number of prediction locations
    params = dict(theta = thetaopt, sigma = 1.0)
    CovParam = lambda *Zvals: CovM(*Zvals, kernel = DiffKernel, **params)
    H = CovParam(Zobs, ExtZpred)

    # Final predictions
    ## Lagrangian Kriging [yhat]
    nugget = 1e-10
    I = np.eye(K.shape[0]); nI = nugget * I
    Kinv = np.linalg.solve(K + nI, I)
    Allpred = LagKrig(Kinv, H, v1, ytrue1, U) 
    SimplePred = SimKrig(Kinv, H, ytrue1)

    ## Unpacking, rearranging the constrained predictions
    yxhat1, yyhat1 = np.zeros((2, nL1));
    SKyxhat1, SKyyhat1 = np.zeros((2, nL1));            # Simple Kriging in the extended design space - unconstrained predictions FOR COMPARISON

    yxxhat1, yyyhat1 = np.zeros((2, nL2));
    SKyxxhat1, SKyyyhat1 = np.zeros((2, nL2));          # Simple Kriging in the extended design space - unconstrained predictions FOR COMPARISON

    for _, i in zip(range(0, 2*nL1, 2), range(nL1)):
        if f1flag == 1:                    
            yxhat1[i] = Allpred[_];
            yyhat1[i] = Allpred[_ + 1]; 
            SKyxhat1[i] = SimplePred[_];
            SKyyhat1[i] = SimplePred[_ + 1]; 

    for _, i in zip(range(2*nL1, 2*nL1+2*nL2,  2), range(nL2)):
        if f1flag == 1:
            yxxhat1[i] = Allpred[_]; 
            yyyhat1[i] = Allpred[_ + 1]; 
            SKyxxhat1[i] = SimplePred[_];
            SKyyyhat1[i] = SimplePred[_ + 1]; 

    if mode == 0:
        return yxhat1, yyhat1, yxxhat1, yyyhat1, SKyxhat1, SKyyhat1, SKyxxhat1, SKyyyhat1
    return Allpred

