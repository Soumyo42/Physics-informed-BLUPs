# Matrices.py
import numpy as np
from typing import List, Any, Callable
from src.covderivatives import torchdiff1D, sqexptorch

printline = lambda *string: print(*string, '\n', 100 * '-' , '\n' , 100 * '-') 
def printifmain(string):  
    if __name__ == '__main__':
        print(string)  

# Matrix U
def Umat(coeffs: List[Any],
    _p: int,):
    '''
    Linear differential coefficient matrix

    :coeffs: floats OR non-constant coeffs (known functional) 
    :_p: Number of DiffEq locations   
    
    Matrix U construction
    ===================
    Designed for inputs of the type [f(x1), f''(x1), f(x2), f''(x2),...]^T

    :Output: TWO dim matrix with DiffEq coefficients
    '''
    ncols = _p
    if np.ndim(coeffs) == 0:
        printifmain('Zero dimensional')
        nrows = ncols
        U = coeffs * np.eye(ncols)
        return U
    coeffs = np.array(coeffs.copy())
    if np.ndim(coeffs) == 1:
        printifmain('Single dimensional')
        op_dim = len(coeffs)                # unary or binary or ternary [...] operation
        nrows = op_dim * ncols
        U = np.zeros((nrows, ncols))
        for _ in range(ncols):
            U[op_dim * _ : op_dim * (_ + 1), _] = coeffs
    elif np.ndim(coeffs) == 2:
        printifmain('Multi dimensional')
        n_dim, op_dim = coeffs.shape
        if n_dim != _p: print('# of rows must match # collocation locations (dim = collocation locs * terms in op)'); return None
        nrows = op_dim * ncols
        U = np.zeros((nrows, ncols))
        for _ in range(ncols):
            U[op_dim * _ : op_dim * (_ + 1), _] = coeffs[_, :]
    else: print('Invalid input'); return None
    return U

# Cov Matrices
Kernel = sqexptorch                                                                                # Change kernel here.
DiffKernel = lambda x, y, xindex, yindex, **kwargs: torchdiff1D(x, y, Kernel, xindex, yindex, **kwargs)

def CovFn(*Z: List[Any],
          kernel: Callable = DiffKernel,
          **kwargs,):
    '''
    Covariance computation for a single extended design coordinate.
    
    :param *Z: Description
    :type *Z: List[Any]
    :param kernel: Description
    :type kernel: Callable
    :param kwargs: kernel params
    '''

    Cov = lambda x, y, xindex, yindex: kernel(x, y, xindex, yindex, **kwargs)                 # New function that accounts for the covariance kernel parameters
    _num = len(Z)
    if _num == 1:
        printifmain('Single inputs Z, computing covariance between (Z, Z)')
        loc1, m1 = Z[0]                                                                       # locations
        return Cov(loc1, loc1, m1, m1)
    if _num == 2:
        printifmain('Two inputs (Z, Z\'), computing covariance between (Z, Z\')')
        loc1, m1 = Z[0]                                                                         # locations
        loc2, m2 = Z[1]                                                                         # orders of derivatives
        return Cov(loc1, loc2, m1, m2)
    else: 
        print('Wrong input for CovFn. consider giving only two inputs.')
        return None


def CovM(*Z: List[Any],
         kernel: Callable = DiffKernel,
         **kwargs,):
    '''
    Covariance matrix construction

    :param Z: All locations and orders of derivatives
    :type Z: List[Any]

    Give it one or two args, it will compute the covariances between the respective indices. 

    :return output: 2D Covariance matrix
    '''
    Covfn = lambda x, y, xindex, yindex: kernel(x, y, xindex, yindex, **kwargs)                 # New function that accounts for the covariance kernel parameters
    _num = len(Z)
    if _num == 1:
        printifmain('Single input Z, constructing covariance between (Z, Z)')
        n = Z[0].shape[0]
        K = np.zeros((n,n))
        for _i in range(n):
            for _j in range(n):
                loc1, m1 = Z[0][_i]; loc2, m2 = Z[0][_j]                                                  # Locations and orders of derivative
                K[_i, _j] = Covfn(loc1, loc2, m1, m2)
        return K
    elif _num == 2:
        printifmain('Two input vectors (Z, Z\'), constructing covariance between (Z, Z\')')
        n = Z[0].shape[0]
        q = Z[1].shape[0]
        H = np.zeros((n, q))
        for _i in range(n):
            for _j in range(q):
                loc1, m1 = Z[0][_i]; loc2, m2 = Z[1][_j]                                            # Locations and orders of derivative
                H[_i, _j] = Covfn(loc1, loc2, m1, m2)
        return H
    else: 
        print('Invalid input. Enter ONE or TWO input lists.') 
        return None


def Kmat(ExtZ: List, n: int, U: List, **Params: dict):
    '''
    Kmat for co-Kriging
    
    :param ExtZ: Extended observations (Z, Z^+)
    :type ExtZ: List
    :param n: NUM of true observations (Z)
    :type n: int
    :param U: U matrix
    :type U: 2D matrix
    :param Params: kernel parameters 
    :type Params: dict
    :return 2D Covariance matrix:
    '''
    CovParam = lambda *Zvals: CovM(*Zvals, **Params)  
    K11 = CovParam(ExtZ[:n])                            # Simple Block structure 
    K12 = CovParam(ExtZ[:n], ExtZ[n:]) @ U
    K21 = K12.T
    K22 = U.T @ CovParam(ExtZ[n:]) @ U
    Kplus = np.block([[K11, K12],                       # [Cov(Z,Z) Cov(Z, Z^+) U]
                    [K21, K22]])                        # [U^T Cov(Z^+, Z) U^T Cov(Z^+, Z^+) U]
    return Kplus                           


def Hmat(ExtZ: List[Any], n: int, Pred: List[Any], U: List, **Params: dict):
    '''
    Hmat for Co-Kriging
    
    :param ExtObs: Observation locations and orders
    :type ExtObs: List[Any]
    :param n: NUM true observations
    :type n: int
    :param Pred: Prediction locations and orders
    :type Pred: List[Any]
    :param U: U matrix
    :type U: List
    :param Params: Optimal Kernel parameters
    :type Params: dict
    :return Hplus: Co-Kriging H matrix
    ''' 
    CovParam = lambda *Zvals: CovM(*Zvals, **Params) 
    H1 = CovParam(ExtZ[:n], Pred)
    H2 = U.T @ CovParam(ExtZ[n:], Pred)
    Hplus = np.block([[H1],                             # [Cov(Z, Z^*)]
                    [H2]])                              # [U^T Cov(Z^+, Z^*)]
    return Hplus



# if __name__ == '__main__':
#     printline('Testing U matrix function')
#     output = U(1,5)
#     print(output)
#     printline()
#     output = U([1,2,3], 5)
#     print(output)
#     printline()
#     output = U([[1,2],[1,4]], 3)
#     print(output)
#     printline()
#     output = U([[1,2],[1,4]], 2)
#     print(output)
#     printline('Testing the covariance differentiation function: DiffKernel')
#     print(f'{DiffKernel(1.0, 2.0, 1, 0, thetaval = 2.0, sigmaval = 1.0):.2f}')
#     printline()