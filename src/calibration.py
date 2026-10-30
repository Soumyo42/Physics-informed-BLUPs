# Hyperparameter optimizaton functions
import numpy as np
seed = 20
np.random.seed(seed)

def reparam(type:str = 'invsoftmax', s = 1.0):
    # Taken from https://arxiv.org/pdf/2101.09747
    # invsoftmax
    if type == 'invsoftmax':
        return lambda param: np.log(np.exp(param/s) - 1)
    # Log reparam 
    else:
        return lambda param: np.log(param)
def reparaminv(type:str = 'invsoftmax', s = 1.0):
    # invsoftmax
    if type == 'invsoftmax':
        return lambda paramd: s * np.log(np.exp(paramd) + 1)
    # Log reparam 
    else:
        return lambda paramd: np.exp(paramd)

# Adam optimizer, manually written
def adamopt(obj, bounds, n_iter, alpha=0.01, beta1=0.9, beta2=0.999, eps = 1e-8, stoploss: int = 50, loss: str = 'LOOMSE'):
    history = np.zeros(n_iter) + 1e10
    r_history = np.zeros(n_iter)                                            # To store the theta values of the optimization routine. 
    x = bounds[0] + (bounds[1] - bounds[0]) * np.random.uniform(0, 1)
    x = reparam()(x)
    delta = 1e-6
    # initialize first and second moments
    m = 0.0
    v = 0.0
    for i in range(n_iter):
        rx, rdx = reparaminv()(x), reparaminv()(x + delta) 
        f1 = obj(rx)
        f2 = obj(rdx)
        # values = torch.tensor([f1, f2],)
        values = np.array([f1, f2],dtype=float)
        # print(f'values = {values}')
        coordinates = np.array([rx, rdx],dtype=float)
        # print(f'coordinates = {coordinates}')
        # gradt = np.gradient(values, spacing = coordinates)[0][0]
        gradt = np.gradient(values, coordinates)[0]
        # print(gradt)
        # m(t) = beta1 * m(t-1) + (1 - beta1) * g(t)
        m = beta1 * m + (1.0 - beta1) * gradt
        # v(t) = beta2 * v(t-1) + (1 - beta2) * g(t)^2
        v = beta2 * v + (1.0 - beta2) * gradt**2
        # mhat(t) = m(t) / (1 - beta1(t))
        mhat = m / (1.0 - beta1**(i+1))
        # vhat(t) = v(t) / (1 - beta2(t))
        vhat = v / (1.0 - beta2**(i+1))
        # x(t) = x(t-1) - alpha * mhat(t) / (sqrt(vhat(t)) + eps)
        x = reparam()(rx - (alpha * mhat / (np.sqrt(vhat) + eps)))
    
        # evaluate candidate point
        score = obj(rx)
        history[i] = score
        r_history[i] = rx

        if i >= stoploss and np.abs(history[i] - history[i-stoploss]) <= 1e-3:
              print(f'Early stopping: difference between, |iter {i} - iter {i-stoploss}| = {np.abs(history[i] - history[i-stoploss]):.4f}')
              # report progress
              print('Iter %d/%d - %s: %.3f   lengthscale: %.3f' % (
                  i + 1, n_iter, loss, score, rx)); 
              rxmin = r_history[history==min(history)]

              if np.ndim(rxmin) == 1:
                   rxmin = rxmin[0]                                       # in case, rxmin is a vector, return only the first theta 
              scoremin = history[r_history == rxmin][0]                   # OPTSCORE
              return rxmin, scoremin, history
        # if gradt <= 1e-6:
        #     print(f'Early stopping: gradient = {gradt}'); 
        #     print('Iter %d/%d - %s: %.3f   lengthscale: %.3f' % (
        #                       i + 1, n_iter, loss, score, rx))
        #     return rx, score, history
        # report progress
        print('Iter %d/%d - %s: %.3f   lengthscale: %.3f' % (
            i + 1, n_iter, loss, score,
            rx
        ))
        # print('>%d f(%s) = %.5f' % (i, x, score))
    rxmin = r_history[history==min(history)]
    if np.ndim(rxmin) == 1:
        rxmin = rxmin[0]                                       # in case, rxmin is a vector, return only the first theta 
        # print('>%d f(%s) = %.5f' % (i, x, score)) 
        scoremin = history[r_history == rxmin][0]              # OPTSCORE
    return rxmin, scoremin, history

# Adam optimizer [Ndim.], manually written
def Adam(obj, bounds, n_iter, alpha=0.01, beta1=0.9, beta2=0.999, eps = 1e-8, stoploss: int = 50, loss: str = 'LOOMSE'):
    history = np.zeros(n_iter)
    x = bounds[:, 0] + np.random.uniform(0, 1, len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    x = reparam()(x)
    delta = 1e-6

    # initialize first and second moments
    m = np.zeros(bounds.shape[0])
    v = np.zeros(bounds.shape[0])

    # run the gradient descent updates
    for t in range(n_iter):
        # calculate the gradient.
        g = np.zeros(bounds.shape[0])                       # Initialize the gradient vector
        I = np.eye(bounds.shape[0])                         # identity matrix
        
        rx = reparaminv()(x)
        f1 = obj(rx)                                        # score

        # Gradient computation
        for _ in range(bounds.shape[0]):
            ei = I[_]                                       # mask array
            rdxi = reparaminv()(x+(delta*ei))               # tiny displacement in a particular dimension. 
            f2 = obj(rdxi)
            values = np.array([f1, f2], dtype=float)
            coordinates = np.array([rx[_], rdxi[_]],        # Coordinates in that particular dimension.
                                   dtype=float) 
            g[_] = np.gradient(values, coordinates)[0]      # gradient component _ at x OR rx.

        # m(t) = beta1 * m(t-1) + (1 - beta1) * g(t)
        m = beta1 * m + (1.0 - beta1) * g
        # v(t) = beta2 * v(t-1) + (1 - beta2) * g(t)^2
        v = beta2 * v + (1.0 - beta2) * g**2
        # mhat(t) = m(t) / (1 - beta1(t))
        mhat = m / (1.0 - beta1**(t+1))
        # vhat(t) = v(t) / (1 - beta2(t))
        vhat = v / (1.0 - beta2**(t+1))
        
        # x(t) = x(t-1) - alpha * mhat(t) / (sqrt(vhat(t)) + eps)
        x = reparam()(rx - (alpha * mhat / (np.sqrt(vhat) + eps)))
        
        # evaluate candidate point
        score = f1
        history[t] = score
        if t >= stoploss and np.abs(history[t] - history[t-stoploss]) <= 1e-3:
            print(f'Early stopping: difference between, |iter {t} - iter {t-stoploss}| = {np.abs(history[t] - history[t-stoploss]):.4f}')
            # report progress
            print(f'Iter {t}/{n_iter} -     {loss}:{score:.2f}     lengthscale:{rx.round(2)}'); return rx, score, history
        # elif (g <= 1e-3).all():
        #     print(f'Early stopping: GRADIENT = {g.round(2)}'); return rx, score, history
        # report progress
        print(f'Iter {t}/{n_iter} -     {loss}:{score:.2f}     lengthscale:{rx.round(2)}')
    return rx, score, history

#################################################################
# 1D LOOMSE functions
#################################################################
def LOOMSESK1D(theta, y, Kmatrix, nugget = 1e-10, **sigmakwargs):
        kernelkwargs = dict(thetaval = theta, **sigmakwargs)
        K = Kmatrix(kernelkwargs)
        N = K.shape[0]
        Kinv = np.linalg.solve(K + (np.eye(N) * nugget), np.eye(N))
        yobs = y.copy().squeeze()
        GAMMA = Kinv @ yobs
        DIAGK = np.diag((np.ones(N)/np.diag(Kinv))**2)
        loomse = 1/N * (GAMMA.T @ DIAGK @ GAMMA)
        return loomse

def LOOVARSK1D(y, Kmatrix, kernelkwargs, nugget = 1e-10):
    K = Kmatrix(kernelkwargs)
    N = K.shape[0]
    Kinv = np.linalg.solve(K + (np.eye(N) * nugget), np.eye(N))
    yobs = y.copy().squeeze()
    ###############################################################
    ###############################################################
    # LOO var criterion optimal sigma
    GAMMA = Kinv @ yobs
    DIAGK = np.diag((np.ones(N)/np.diag(Kinv)))
    sigma = 1/N * (GAMMA.T @ DIAGK @ GAMMA)
    ###############################################################
    print(f'sigma^2 = {sigma:.2f}')
    sigmaopt = np.sqrt(sigma)
    return sigmaopt

def LOOMSELK1D(theta, y, X, Kmatrix, Umatrix, Hmatrix, Kfn, LK, nugget = 1e-10, **sigmakwargs):
        kernelkwargs = dict(thetaval = theta, **sigmakwargs)
        MSELK = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            mask = np.ones(len(X), dtype = bool)
            mask[i] = False
            Obs = X[mask]
            yobs = y[mask]
            Pred = np.array([X[i]])
            # U matrix
            step = 1
            q = 3 * Pred.shape[0]       # 2 * Primeobs + Pred
            shape = (q, 1)
            U = Umatrix(shape,
                        step,
                        1.0,
                        1.0)
            v = np.zeros(Pred.shape[0])                                      # RHS of the constraint equations
            K = Kmatrix(Obs, kernelkwargs)
            H = Hmatrix(Obs, Pred, Pred, kernelkwargs)
            N = K.shape[0]
            Kinv = np.linalg.solve(K + (np.eye(N) * nugget), np.eye(N))
            args = (K,H,U,v,yobs,Pred,Kfn,1e-10,0.0)
            Ystarsi1e2, ypred, Yprimeprimestarsi1e2, sdsi1e2 = LK(*args, sigma = sigmakwargs['sigmaval'], theta = theta)
            MSELK[i] = (ypred - y[i])**2
        loomse = MSELK.mean()
        return loomse

def LOOMSECK1D(theta, y, Nobs, Kmatrix, nugget = 1e-10, **sigmakwargs):
        kernelkwargs = dict(thetaval = theta, **sigmakwargs)
        K = Kmatrix(kernelkwargs)
        N = K.shape[0]
        Kinv = np.linalg.solve(K + (np.eye(N) * nugget), np.eye(N))
        yobs = y.copy().squeeze()
        GAMMA = Kinv @ yobs
        DIAGK = np.diag((np.ones(N)/np.diag(Kinv))**2)
        # Sieve out the true observations ONLY.
        DIAG = np.zeros((N,N))
        DIAG[:Nobs, :Nobs] = np.eye(Nobs)
        loomse = 1/N * (GAMMA.T @ DIAGK @ DIAG @ GAMMA)
        return loomse

def LOOVARCK1D(y, Nobs, Kmatrix, kernelkwargs, nugget = 1e-10):
        K = Kmatrix(kernelkwargs)
        N = K.shape[0]
        Kinv = np.linalg.solve(K + (np.eye(N) * nugget), np.eye(N))
        yobs = y.copy().squeeze()
        ###############################################################
        ###############################################################
        # LOO var criterion optimal sigma
        GAMMA = Kinv @ yobs
        DIAGK = np.diag((np.ones(N)/np.diag(Kinv)))
        DIAG = np.zeros((N,N))
        DIAG[:Nobs, :Nobs] = np.eye(Nobs)
        sigmasq = 1/N * (GAMMA.T @ DIAGK @ DIAG @ GAMMA)
        ###############################################################
        print(f'sigma^2 = {sigmasq:.4f}')
        sigmaopt = np.sqrt(sigmasq)
        return sigmaopt
#######################################################################
#######################################################################
# 2D LOOMSE functions
#######################################################################
def LOOMSESK2D(theta, y, Kmatrix, nugget = 1e-10, **sigmakwargs):
        kernelkwargs = dict(theta = theta, **sigmakwargs)
        K = Kmatrix(kernelkwargs)
        N = K.shape[0]
        Kinv = np.linalg.solve(K + (np.eye(N) * nugget), np.eye(N))
        yobs = y.copy().squeeze()
        GAMMA = Kinv @ yobs
        DIAGK = np.diag((np.ones(N)/np.diag(Kinv))**2)
        loomse = 1/N * (GAMMA.T @ DIAGK @ GAMMA)
        return loomse

def LOOVARSK2D(y, Kmatrix, kernelkwargs, nugget = 1e-10):
        K = Kmatrix(kernelkwargs)
        N = K.shape[0]
        Kinv = np.linalg.solve(K + (np.eye(N) * nugget), np.eye(N))
        yobs = y.copy().squeeze()
        ###############################################################
        ###############################################################
        # LOO var criterion optimal sigma
        GAMMA = Kinv @ yobs
        DIAGK = np.diag((np.ones(N)/np.diag(Kinv)))
        sigma = 1/N * (GAMMA.T @ DIAGK @ GAMMA)
        ###############################################################
        print(f'sigma^2 opt. = {sigma}')
        sigmaopt = np.sqrt(sigma)
        return sigmaopt

def LOOMSECK2D(theta, y, Nobs, Kmatrix, nugget = 1e-9, **sigmakwargs):
        kernelkwargs = dict(theta = theta, **sigmakwargs)
        K = Kmatrix(kernelkwargs)
        N = K.shape[0]
        Kinv = np.linalg.solve(K + (np.eye(N) * nugget), np.eye(N))
        yobs = y.copy().squeeze()
        GAMMA = Kinv @ yobs
        DIAGK = np.diag((np.ones(N)/np.diag(Kinv))**2)
        # Sieve out the true observations ONLY.
        DIAG = np.zeros((N,N))
        DIAG[:Nobs, :Nobs] = np.eye(Nobs) 
        loomse = 1/Nobs * (GAMMA.T @ DIAGK @ DIAG @ GAMMA)
        # Full data LOOMSE
        # loomse = 1/N * (GAMMA.T @ DIAGK @ GAMMA)
        return loomse

def LOOVARCK2D(y, Nobs, Kmatrix, kernelkwargs, nugget = 1e-10):
        K = Kmatrix(kernelkwargs)
        N = K.shape[0]
        Kinv = np.linalg.solve(K + (np.eye(N) * nugget), np.eye(N))
        yobs = y.copy().squeeze()
        ###############################################################
        # NLL optimla sigma
        # sigma = (1/N) * Zobs.T @ Kinv @ Zobs
        ###############################################################
        # LOO var criterion optimal sigma
        GAMMA = Kinv @ yobs
        DIAGK = np.diag((np.ones(N)/np.diag(Kinv)))
        DIAG = np.zeros((N,N))
        DIAG[:Nobs, :Nobs] = np.eye(Nobs)
        sigma = 1/Nobs * (GAMMA.T @ DIAGK @ DIAG @ GAMMA)
        # sigma = 1/N * (GAMMA.T @ DIAGK @ GAMMA)
        ###############################################################
        print(f'sigma^2 opt. = {sigma:.2f}')
        sigmaopt = np.sqrt(sigma)
        return sigmaopt