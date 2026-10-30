# Simple Kriging for potential flow prediction.
# Imports 
import numpy as np
from numpy import cos, sin, exp 
import matplotlib.pyplot as plt
from src.matrices import CovM, Umat
from src.covderivatives2D import AnalyticalSqExpdiff2D
from scipy.interpolate import griddata
from src.calibration import adamopt, LOOMSESK2D, LOOVARSK2D
import time 

# start = time.time()                                         # To compute the full run time
###################

# Constants
PI = np.pi
seed = 1
np.random.seed(seed)                                        # Reproducibility

# user inputs
xl = 0; xr = 5
yd = 0; yu = 5
xinf = 0                                                    # x position of infinity
nobs = 5                                                    # Observed velocity/potential function values (before the obstacle)
ndomobs_r = nobs                                            # Domain observations (after obstacle)  
ndomobs_m = 2                                               # Domain observations (in the middle) 
npred = 20                                                  # discretization for prediction points.
hypropt = 0                                                 # For hyperparameter optimization
UQflag = 1                                                  # Uncertainty quantification
figsave = 1                                                 # To save the results
IMGPATH = r'IMAGES/'                                        # Image path

# circular obstacle 
r: float = 1
centre: tuple = ((xr + xl)/2,
                  (yd + yu)/2)

## function to determine whether a point is outside circular obstacle. 
def isOutsideCircle(point: tuple,
                    centre: tuple = centre,
                    radius: float = r):
    d = np.linalg.norm(point - centre)
    if d <= radius:
        return False
    return True

isoutkwargs = dict(centre = centre, radius = r)             # dict. of constants to pass to isOutsideCircle  

# Kernel hyperparameters
sigma = 0.775
theta = 2.10                                                # Default, SK opt. 2.10, 0.775  
kwargs = dict(sigma = sigma, theta = theta)                 # dict. of kernel params
vinfx: float = 1.0                                          # Velocity_x at infinity
vinfy: float = 0                                            # velocity_y at infinity 

# infinity observations
## Initialization: size = (NUM X COORDS/DERIVATIVE ORDERS X [x, y]/[xindex, yindex]).
## Observation locations arranged as: -inf locations ---- middle locations ---- +inf locations
##                                          nobs     ----    ndomobs_m     ----    ndomobs_r
Obs = np.zeros((2 * (nobs + ndomobs_m + ndomobs_r), 2, 2))  # observation locations N * DIM = 2
e1 = nobs; e2 = e1 + ndomobs_m; 
e3 = ndomobs_r + e2                                         # Useful checkpoints. 
xvals = xl + xinf + np.zeros(nobs)                          # observation locations at -infinity (x-coordinates)
yvals = np.linspace(yd, yu, nobs)                           # observation locations (y-coordinates)
Obs[:e1, 0, 0] = xvals; Obs[:e1, 0, 1] = yvals              # vx at -inf. m = (1,0)
xvals = xr + np.zeros(ndomobs_r);                           # observation locations at +infinity (x-coordinates)
Obs[e2:e3, 0, 0] = xvals; Obs[e2:e3, 0, 1] = yvals          # vx at +inf. m = (1,0)
xvals = np.array(2 * [(xr - xl)/2]);                        # MID domain observation locations (to ensure flow doesn't exit domain)
yvals = np.array([yd, yu])                                  # x-y components
Obs[e1:e2, 0, 0] = xvals; Obs[e1:e2, 0, 1] = yvals          # vx at middle points m = (1,0)
Obs[e3:, 0, :] = Obs[:e3, 0, :]                             # Repeat the same observation locations for vy, m = (0,1)
Obs[:e3, 1, 0] = 1; Obs[e3:, 1, 1] = 1                      # Setting the right order of derivatives: m=(1,0) & m = (0,1). 

# Prediction points
xvals = np.linspace(xl, xr, npred)                          # prediction locations x-components
yvals = np.linspace(yd, yu, npred)                          # prediction locations y-components
xvals, yvals = np.meshgrid(xvals, yvals)
Pred = np.zeros((npred**2, 2, 2))                           # Initializing the prediction array
Pred[:, 0, 0] = xvals.flatten();
Pred[:, 0, 1] = yvals.flatten();

OutsideCircle = lambda COORDS: isOutsideCircle(COORDS,      # New function that uses the centre and radius information 
                                                **isoutkwargs) 
OutsideCoords = np.array(list(filter(OutsideCircle, 
                                    Pred[:,0,:])))
q = OutsideCoords.shape[0]                                  # number of prediction locations
Pred = np.zeros((2*q, 2, 2))                                # 2*NUM X COORDS/DERIVATIVES X [x,y]/[xindex, yindex]
Pred[:q, 0, :] = OutsideCoords                              # Outside circle prediction points, vx
Pred[q:, 0, :] = OutsideCoords                              # Outside circle prediction points, vy
Pred[:q, 1, 0] = 1; Pred[q:, 1, 1] = 1                      # m = (1,0); m = (0,1) 

## Analytical solution: https://en.wikipedia.org/wiki/Potential_flow_around_a_circular_cylinder (solution in polar coords)

## cos component of boundary normal vector
def Bdcos(Coords: np.ndarray,
          centre: tuple):
    if np.ndim(Coords) == 1:
        x = Coords[0] - centre[0]
        y = Coords[1] - centre[1]
    elif np.ndim(Coords) == 2:
        x = Coords[:, 0] - centre[0] 
        y = Coords[:, 1] - centre[1]
    dist = np.sqrt(x**2 + y**2)
    return x/dist

## sin component of boundary normal vector
def Bdsin(Coords: np.ndarray,
          centre: tuple):
    if np.ndim(Coords) == 1:
        x = Coords[0] - centre[0]
        y = Coords[1] - centre[1]
    elif np.ndim(Coords) == 2:
        x = Coords[:, 0] - centre[0] 
        y = Coords[:, 1] - centre[1]
    dist = np.sqrt(x**2 + y**2)
    return y/dist

U = np.sqrt(vinfx**2 + vinfy**2)
n = Obs.shape[0]; obs_loc = int(n/2)                        # Observation locations is 0.5 * # of vectors. 
COS = Bdcos(Obs[:obs_loc, 0, :], centre)
SIN = Bdsin(Obs[:obs_loc, 0, :], centre)
Vactualobs = np.zeros((obs_loc, 2))
for i in range(obs_loc): 
    M = np.array([[COS[i], -SIN[i]],                        # Matrix to convert from polar coords ---> cartesian 
                [SIN[i], COS[i]]])

    r2R2 = (r**2)/np.linalg.norm(Obs[i,0,:] - centre)
    Vr = U * (1 - r2R2) * COS[i]
    Vt =  -U * (1 + r2R2) * SIN[i]
    Vactualobs[i] = M @ np.array([Vr, Vt])
Vobs = Vactualobs                                           # Analytical soln ---> true observations for the model.

# Kernel differentiation function 
DiffKernel = lambda x, y, xindex, yindex, **kwargs: AnalyticalSqExpdiff2D(x, y,
                                                                          xindex, yindex,
                                                                          **kwargs)
#######################################
# LOOCV optimization steps
# #####################################
ytrue = np.zeros(n)                                         # Full vector of observations.
ytrue = Vobs.flatten(order = 'F');                          # vx obs first and then vy obs. 

## Matrix K
Kmatfn = lambda params: CovM(Obs, kernel=DiffKernel, **params)

if hypropt == 1:
    bL = 0.5; bR = 1.0                                      # left and right bounds
    bounds = [bL, bR]                                       # To initialize theta (correlation parameter)
    n_iter = 1000
    loomsefn = lambda theta: LOOMSESK2D(theta, ytrue, Kmatfn, sigma = 1.0)
    thetaopt, optscore, history = adamopt(loomsefn, bounds, n_iter = n_iter)
    print(f'{thetaopt=:.2f}'); print(f'{optscore=:.4f}')
    plt.plot(range(n_iter), history); plt.savefig(IMGPATH+'lossfluidSK'); plt.show()      # To visualize LOOCV loss curve
    sigmaopt = LOOVARSK2D(ytrue, Kmatfn, dict(theta = thetaopt, sigma = 1.0))
    print(f'{sigmaopt=:.2f}')
else: 
    print(f'Using the fixed kernel params (user inputs): {theta=:.2f}, {sigma=:.2f}')
    thetaopt = theta; sigmaopt = sigma

## Matrix K (optimal)
K = Kmatfn(dict(theta = thetaopt,                           # LOOCV Optimal Kplus matrix 
                     sigma = 1.0))

## Matrix H
params = dict(theta = thetaopt, sigma = 1.0)            
H = CovM(Obs, Pred, kernel = DiffKernel, **params)

# Final predictions
## Simple Kriging [yhat]
nugget = 1e-10
I = np.eye(K.shape[0]); nI = nugget * I
Kinv = np.linalg.solve(K + nI, I)
yhat = H.T @ Kinv @ ytrue

if UQflag == 1:
    ## Uncertainty quantification: simple Kriging expression
    BaseVar = 1e-5                                          # To make sure Conditional variance estimate is positive. 
    Kpred = CovM(Pred, kernel = DiffKernel, **params)       # Unconditional variance
    CondVar = Kpred - (H.T @ Kinv @ H)                      # Minimal MSE
    CondVar += BaseVar                                      # Small quantity added to make it positive
    CondSigma = np.sqrt(sigmaopt**2 * np.diag(CondVar));

# Visualization
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12})       
widf = 1.5

## Visualizing the experiment design space.
plt.figure(figsize = (12, 10))
plt.cla()
plt.grid()
plt.scatter(Obs[:, 0, 0], 
            Obs[:, 0, 1],
            marker = 'x', c = 'r',
            label = 'Observations')
plt.plot(centre[0] + r * cos(np.linspace(0, 2 * PI, 100)),
         centre[1] + r * sin(np.linspace(0, 2 * PI, 100)),
         alpha = 0.5, c = 'grey', label = 'Obstacle')
plt.title('Experiment design (locations)')
plt.legend()
plt.show()

## Co-Kriging flow prediction
plt.figure(figsize = (12, 10))
plt.cla()
quiverkwargs = dict(scale = 1/0.05, width =(xr - xl)/(widf*1e3), pivot = 'tail')
Q1 = plt.quiver(Obs[:obs_loc, 0, 0], Obs[:obs_loc, 0, 1],
                Vobs[:, 0], Vobs[:, 1],
                color = 'red', alpha = 1,
                **quiverkwargs) 
Q2 = plt.quiver(Pred[:q, 0, 0], Pred[:q, 0, 1],
                yhat[:q], yhat[q:],
                np.hypot(yhat[:q], yhat[q:]), 
                alpha = 0.5,
                **quiverkwargs)
plt.quiverkey(Q1, 1.02, 1.05, 1, label='Observations', labelpos = 'E')
plt.quiverkey(Q2, 1.02, 1.03, 1, label='Predictions', labelpos = 'E')
plt.plot(centre[0] + r * cos(np.linspace(0, 2 * PI, 100)),
             centre[1] + r * sin(np.linspace(0, 2 * PI, 100)),
             alpha = 0.5, c = 'grey', label = 'Obstacle')
plt.colorbar()
plt.title(f'Simple Kriging: {thetaopt=:.2f}, {sigmaopt=:.4f}')
if figsave == 1:
    plt.savefig(IMGPATH+'FluidSK.pdf', dpi = 300, bbox_inches = 'tight', pad_inches = 0)
plt.show()

## Simple Kriging uncertainty in prediction.
## Generalized chi^2 distribtion. Quadratic form of two dependent gaussian vectors vx, vy. 
if UQflag == 1:
    CondVar = sigmaopt**2 * CondVar;
    CVxVx = np.diag(CondVar[:q, :q])                              # Var[Vx]
    CVxVy = np.diag(CondVar[:q, q:])                              # Cov[Vx, Vy]
    CVyVy = np.diag(CondVar[q:, q:])                              # Var[Vy]
    ChiVar = np.zeros(q)                                          # Chi^2 params.
    for i in range(q):
        Mu = np.array([yhat[i],                                   # predicted Vx 
                       yhat[i + q]])                              # predicted Vy
        Cov = np.array([[CVxVx[i], CVxVy[i]],                     # Cov matrix for point prediction
                        [CVxVy[i], CVyVy[i]]])
        eigvect, eigmatrix = np.linalg.eigh(Cov)
        b = eigmatrix @ Mu
        ChiVar[i] = np.sum(2 * eigvect**2 * (1 + 2*b**2))         # A derivation can be found in https://math.stackexchange.com/a/442916
                                                                  # A paper about normal ---> generalized chi^2: https://arxiv.org/pdf/2012.14331
    plt.figure(figsize = (12,10))
    plt.cla()
    X, Y = np.meshgrid(np.linspace(xl, xr, npred),
                        np.linspace(yd, yu, npred))
    UQ = griddata(Pred[:q,0,:], ChiVar,                           # since we only have uncertainty values around the obstacle, we need to interpolate on a grid to use contourf.
                  (X,Y), method = 'linear', 
                  fill_value = 0)
    plt.contourf(X, Y, 2*np.sqrt(UQ).reshape(npred, npred), 
                 vmin = 0.0, vmax = 1.0)                          # Same scale colorbar 
    plt.clim(0.0, 0.200)                                          # Same scale colorbar
    plt.colorbar()
    plt.fill(centre[0] + r * cos(np.linspace(0, 2 * PI, 100)),
             centre[1] + r * sin(np.linspace(0, 2 * PI, 100)),
             alpha = 1, c = 'white', label = 'Obstacle')
    plt.title(f'Simple Kriging UQ: {thetaopt=:.2f}, {sigmaopt=:.4f}')
    if figsave == 1:
        plt.savefig(IMGPATH+'FluidSKUQ.pdf', dpi = 300, bbox_inches = 'tight', pad_inches = 0)
    plt.show()
