# Lagrangian Kriging for 2D potential flow prediction / reconstruction
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
nrandobs = 0                                                # Random observations in the domain
nb = 10                                                     # Collocation points around the obstacle boundary
nd = 10                                                     # Collocation points in the domain (Continuity equtaion = Laplacian zero).
npred = 20                                                  # discretization for prediction points.
hypropt = 0                                                 # For hyperparameter optimization
UQflag = 1                                                  # Uncertainty quantification
figsave = 1                                                 # To save figure
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

# Kernel hyperparameters
sigma = 3.24
theta = 2.01                                                # Default, LK opt. 2.01, 3.24 
thetaInterpol = 1.01                                        # Interpolation length scale
sigmaInterpol = 1.0                                         # Interpolating boundary predictions of Lagrangian Kriging
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

# Prediction points (only at the boundray)
OutsideCircle = lambda COORDS: isOutsideCircle(COORDS,      # New function that uses the centre and radius information 
                                               **isoutkwargs) 
BdTheta = np.linspace(0, 2 * PI, nb)                        # Discretizing by angle. 
B = np.zeros((2*nb, 2, 2));                                 # Initializing array of Boundary collocation points. NUM * DIM=2 X COORDS X DERIVATIVE ORDER
coscoeffs = np.zeros(nb); sincoeffs = np.zeros(nb) 
for _, i in zip(range(0, 2*nb, 2), range(nb)):
    angle = BdTheta[i]                                      # iterating over angles
    B[_, 0, 0] = centre[0] + r * cos(angle)                 # prediction locations x-component. 
    B[_, 0, 1] = centre[1] + r * sin(angle)                 # prediction location y-component.
    B[_+1, 0, 0] = centre[0] + r * cos(angle)               # prediction locations x-component.
    B[_+1, 0, 1] = centre[1] + r * sin(angle)               # prediction location y-component.
    B[_, 1, 0] = 1; B[_+1, 1, 1] = 1                        # RIGHT ORDER OF DERIVATIVES (1,0) & (0,1)
    coscoeffs[i] = Bdcos(B[_,0,:], centre)
    sincoeffs[i] = Bdsin(B[_,0,:], centre)                  # boundary normal vector coordinates at B locations.

## Analytical solution: https://en.wikipedia.org/wiki/Potential_flow_around_a_circular_cylinder (solution in polar coords)
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

# Matrix U
U = Umat(np.stack([coscoeffs, sincoeffs],                   # null normal component. cos(P) dx phi + sin(P) dy phi
                   axis = -1),                              # Stack along a new dimension (axis = -1 => new dimension = last dimension)
                   nb) 

# Kernel differentiation function 
DiffKernel = lambda x, y, xindex, yindex, **kwargs: AnalyticalSqExpdiff2D(x, y,
                                                                          xindex, yindex,
                                                                          **kwargs)
#######################################
## LOOCV optimization steps
######################################
v = np.zeros(nb)                                            # RHS of DiffEqns
n = Obs.shape[0];
ytrue = np.zeros(n)                                         # Full vector of observations.
ytrue = Vobs.flatten(order = 'F');                          # vx obs first and then vy obs. 

## Matrix K
Kmatfn = lambda params: CovM(Obs, kernel = DiffKernel, **params)

if hypropt == 1:
    bL = 0.5; bR = 1.0                                      # left and right bounds
    bounds = [bL, bR]                                       # To initialize theta (correlation parameter)
    n_iter = 1000
    loomsefn = lambda theta: LOOMSESK2D(theta, ytrue, Kmatfn, sigma = 1.0)
    thetaopt, optscore, history = adamopt(loomsefn, bounds, n_iter = n_iter)
    print(f'{thetaopt=:.2f}'); print(f'{optscore=:.4f}')
    plt.plot(range(n_iter), history); plt.savefig(IMGPATH+'lossfluidLK'); plt.show()      # To visualize LOOCV loss curve
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
CovParam = lambda *Zvals: CovM(*Zvals, kernel = DiffKernel, **params)
H = CovParam(Obs, B)

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
    Kstar = CovM(Zpred, kernel = DiffKernel, 
                        theta = theta,                  # Unconditional variance.
                        sigma = sigma) 
    M1 = np.outer(ytrue, lamda) @ U.T
    M2 = (H + M1).T @ Kinv @ (M1 - H)                   # LK contribution
    Var = np.diag(Kstar + M2)
    BaseVar = 1e-5                                      # To make sure Conditional variance estimate is positive. 
    return np.sqrt(Var + BaseVar)  
##########################################
# Final predictions
## Lagrangian Kriging [yhat]
nugget = 1e-10
I = np.eye(K.shape[0]); nI = nugget * I
Kinv = np.linalg.solve(K + nI, I)
Allpred = LagKrig(Kinv, H, v, ytrue, U)

if UQflag == 1:
    ## Uncertainty quantification: simple Lagrangian Kriging expression
    AllCondSigma = LagVar(B, Kinv, H, v, ytrue, U)

vxhat, vyhat = np.zeros((2, nb));
sigx, sigy = np.zeros((2, nb));
for _, i in zip(range(0, 2*nb, 2), range(nb)):
    vxhat[i] = Allpred[_]                               # Vx predictions
    vyhat[i] = Allpred[_+1]                             # Vy predictions
    if UQflag == 1:
        sigx[i] = AllCondSigma[_]
        sigy[i] = AllCondSigma[_+1]

## Simple Kriging based interpolation
## Prediction points
xvals = np.linspace(xl, xr, npred)                          # prediction locations x-components
yvals = np.linspace(yd, yu, npred)                          # prediction locations y-components
xvals, yvals = np.meshgrid(xvals, yvals)
Pred = np.zeros((npred**2, 2, 2))                           # Initializing the prediction array
Pred[:, 0, 0] = xvals.flatten();
Pred[:, 0, 1] = yvals.flatten();
OutsideCoords = np.array(list(filter(OutsideCircle, 
                                    Pred[:, 0, :])))
q = OutsideCoords.shape[0]                                  # number of prediction locations
Pred = np.zeros((q, 2, 2))                                  # Order of derivatives kept zero.   
Pred[:, 0, :] = OutsideCoords                               # Outside the circle prediction locations 

TotObs = np.zeros((obs_loc + nb,                            # Total observations locations: considering LK predictions as true observations 
                   2, 2))                                   # Order of derivatives kept zero since we are using the undifferentiated kernel.                                   
TotObs[:obs_loc, 0, :] = Obs[:obs_loc,0,:]                  # Original observation locations
TotObs[obs_loc:, 0, :] = B[::2,0,:]                         # Boundary prediction locations                                                 
Vvect = np.vstack([*Vobs.flatten(), *Allpred])              # Combined observations. Vobs arranged in vx-vy criss-cross arrangement and Allpred has the same arrangement already. 

## Interpolation K matrix
def Kinterpol(params):
    K = CovM(TotObs, kernel = DiffKernel, **params)
    Kkron = np.kron(K, np.eye(2))                           # K matrix with zero cross-covariances between vx and vy. Assuming criss-cross arrangement of vx and vy vectors.
    return Kkron

if hypropt == 1:
    bL = 0.5; bR = 1.0                                      # left and right bounds
    bounds = [bL, bR]                                       # To initialize theta (correlation parameter)
    n_iter = 1000
    loomsefn = lambda theta: LOOMSESK2D(theta, Vvect, Kinterpol, sigma = 1.0)
    thetaopti, optscore, history = adamopt(loomsefn, bounds, n_iter = n_iter)
    print(f'{thetaopti=:.2f}'); print(f'{optscore=:.4f}')
    plt.plot(range(n_iter), history); plt.show()            # To visualize LOOCV loss curve
    sigmaopti = LOOVARSK2D(Vvect, Kinterpol, dict(theta = thetaopti, sigma = 1.0))
    print(f'{sigmaopti=:.2f}')
else: 
    print(f'Using the fixed kernel params (user inputs): {theta=:.2f}, {sigma=:.2f}')
    thetaopti = thetaInterpol; sigmaopti = sigmaInterpol

## Optimal interpolation matrices.
params = dict(theta = thetaopti, sigma = sigmaopti)          # Optimal params
Kkron = Kinterpol(params)
H = CovM(TotObs, Pred, kernel = DiffKernel, **params)
Hkron = np.kron(H, np.eye(2))

# Interpolated predictions
nugget = 1e-10
I = np.eye(Kkron.shape[0]); nI = nugget * I
Kkroninv = np.linalg.solve(Kkron + nI, I)
Vstarkron = Hkron.T @ Kkroninv @ Vvect                     # Simple Kriging predictions
Vstarkron = Vstarkron.reshape(-1,2)                        # New axis indicating Vx, Vy

# Visualizations
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12})       
widf = 1.5

plt.figure(figsize = (12, 10))
plt.cla()
plt.grid()
plt.scatter(Obs[:, 0, 0], 
            Obs[:, 0, 1],
            marker = 'x', c = 'r',
            label = 'Observations')
plt.scatter(B[:, 0, 0],
            B[:, 0, 1],
            marker = '*', c = 'green', 
            label = 'Boundary collocation points')
plt.plot(centre[0] + r * cos(np.linspace(0, 2 * PI, 100)),
         centre[1] + r * sin(np.linspace(0, 2 * PI, 100)),
         alpha = 0.5, c = 'grey', label = 'Obstacle')
plt.title('Experiment design (locations)')
plt.legend()
plt.show()


## Lagrangian Kriging predictions
plt.figure(figsize = (12, 10))
plt.cla()
quiverkwargs = dict(scale = 1/0.05, width =(xr - xl)/(widf*1e3), pivot = 'tail')
Q1 = plt.quiver(Obs[:obs_loc, 0, 0], Obs[:obs_loc, 0, 1],
                Vobs[:, 0], Vobs[:, 1],
                color = 'red', alpha = 1,
                **quiverkwargs) 
Q2 = plt.quiver(B[::2, 0, 0], B[::2, 0, 1],                        # To access all locations, we need to slice through the array with a step size = 2 (since locations repeat twice)
                vxhat, vyhat,
                np.hypot(vxhat, vyhat),
                **quiverkwargs)
plt.quiverkey(Q1, 1.02, 1.05, 1, label='Observations', labelpos = 'E')
plt.quiverkey(Q2, 1.02, 1.03, 1, label='Predictions', labelpos = 'E')
plt.plot(centre[0] + r * cos(np.linspace(0, 2 * PI, 100)),
             centre[1] + r * sin(np.linspace(0, 2 * PI, 100)),
             alpha = 0.5, c = 'grey', label = 'Obstacle')
plt.colorbar()
plt.title(f'Lagrangian Kriging: {thetaopt=:.2f}, {sigmaopt=:.4f}')
if figsave == 1:
    plt.savefig(IMGPATH+'FluidLK.pdf', dpi = 300, bbox_inches = 'tight', pad_inches = 0)
plt.show()

## Interpolated predictions
plt.figure(figsize = (12, 10))
plt.cla()
quiverkwargs = dict(scale = 1/0.05, width =(xr - xl)/(widf*1e3), pivot = 'tail',
                    clim = (0.0, 1.7))                             # Colorbar
Q1 = plt.quiver(Obs[:obs_loc, 0, 0], Obs[:obs_loc, 0, 1],
                Vobs[:, 0], Vobs[:, 1],
                color = 'red', alpha = 1,
                **quiverkwargs) 
Q2 = plt.quiver(B[::2, 0, 0], B[::2, 0, 1],                        # To access all locations, we need to slice through the array with a step size = 2 (since locations repeat twice)
                vxhat, vyhat,
                color = 'blue',
                **quiverkwargs)
Q2 = plt.quiver(Pred[:, 0, 0], Pred[:, 0, 1],
                Vstarkron[:,0], Vstarkron[:,1],
                np.hypot(Vstarkron[:,0], Vstarkron[:,1]), 
                alpha = 0.5,
                **quiverkwargs)
plt.plot(centre[0] + r * cos(np.linspace(0, 2 * PI, 100)),
             centre[1] + r * sin(np.linspace(0, 2 * PI, 100)),
             alpha = 0.5, c = 'grey', label = 'Obstacle')
plt.colorbar()
plt.title(f'Interpolation: {thetaopti=:.2f}, {sigmaopti=:.4f}')
if figsave == 1:
    plt.savefig(IMGPATH+'FluidLKinterpol.pdf', dpi = 300, bbox_inches = 'tight', pad_inches = 0)
plt.show()