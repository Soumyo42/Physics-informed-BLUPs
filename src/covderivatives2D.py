
# imports
import numpy as np
import random 
from sympy import exp, diff, symbols
from torch import sum as torchsum
from torch import exp as torchexp
from torch import tensor as tensor
from torch.autograd import grad as torchgrad
from torch import norm as torchnorm

x1, x2, y1, y2, sigma, theta = symbols('x1 x2 y1 y2 c s')
sqexp2D = sigma**2 * exp(-0.5 * ((x1 - y1)**2 + (x2 - y2)**2)
                                                /theta**2)

vars = [x1, x2, y1, y2]

# Multi-index to individual components 
def multi_index_to_x(index: list) -> list:
    """
    Takes index [l1,...,ld, ld+1,...,l2d] --> [l1 * 0, l2 * 1, ..., l2d * 2d-1]

    In words: Takes the multi-index for and gives how many times we differentiate by each vars. 

    i.e. multi-index [2,0,1,3] --> [0,0,2,3,3,3]
    """
    N = len(index)
    order = [] 
    i = 0
    copy = index.copy()                                                     # Doing any operations on the input list means the list changes everywhere in the program. If we were to use the same list another time, it would be empty. Hence we copye.
    while i < N:
        if bool(copy[i]):
            order.append(i) 
            copy[i] -= 1
            i = 0
        else: 
            i += 1
    return order

def sympydiff2D(x: np.ndarray,
                y: np.ndarray,
                kernelexp,
                xindex: list,
                yindex: list,
                sigmaval: 1.0,
                thetaval: 1.0) -> float:
    x1, x2, y1, y2, sigma, theta = symbols('x1 x2 y1 y2 c s')               # Declaring the variables in local scope
    vars = [x1, x2, y1, y2]
    index = [*xindex, *yindex]
    
    subs = {x1: x[0],
            x2: x[1],
            y1: y[0],
            y2: y[1],
            sigma: sigmaval,
            theta: thetaval}
    
    if index == [0,0,0,0]:
        return kernelexp.evalf(subs = subs)                           
    order = multi_index_to_x(index)
    diffvar = [vars[o] for o in order]
    ans  = diff(kernelexp, *diffvar).evalf(subs = subs)
    return ans

# Analytical implementation for 2D Covariance function
# To check for a list --> number condition (in the case of first-derivative)
def check_num(to_check: list): 
    if len(to_check) == 1:
        return to_check[0]

# From index to the particular h function
def index_to_h(x1: np.array, x2: np.array, idx: int):
    h = 2 * (x1[idx] - x2[idx])
    return h

# From index to the particular hbar function. Only for 2D. change -= 2 with dimension in the general case. 
def index_to_hbar(x1: np.array, x2: np.array, idx: int):
    idx -= 2
    hbar = 2 * (x2[idx] -x1[idx])
    return hbar

# To check the index in multi-index and return the appropriate h or hbar function 
def check_idx(x1: np.array, x2: np.array, idx: int):
    if idx <= 1:
        return index_to_h(x1, x2, idx)
    else: 
        return index_to_hbar(x1, x2, idx)

# delta prime as defined in the document
def delta_prime(i: int, j: int):
    if j - i == 2 or i - j == 2: 
        return 1
    elif j == i: 
        return 1
    else: 
        return 0
    
# delta_prime_prime as defined in the document
def delta_prime_prime(i: int, j: int): 
    if np.abs(i - j) <= 1:
        return 0
    else:
        return 1

# Master function
def AnalyticalSqExpdiff2D(x1: np.array, x2: np.array, xindex: list, yindex: list, sigma: float = 1, theta: float = 1): 
    index = [*xindex, *yindex]
    normsq = np.linalg.norm(x1 - x2) ** 2
    derivative_order = sum(index)
    C = -1/(2 * theta**2)
    sqexp = sigma**2 * np.exp(C * normsq) 
    
    if derivative_order == 0:                                                                                                       # Returns the actual kernel
        return sqexp                                                                                                                    

    if derivative_order == 1:                                                                                                       # Returns the first order derivative
        dx1 = multi_index_to_x(index)
        dx1 = check_num(dx1)                                                                                                        # Checks if dx1 is a number
        return C * check_idx(x1,x2,dx1) * sqexp
    
    if derivative_order == 2:                                                                                                       # Returns the second order derivative
        dx1, dx2 = multi_index_to_x(index)
        term1 = C**2 * check_idx(x1, x2, dx1) * check_idx(x1, x2, dx2)
        term2 = delta_prime(dx1, dx2) * 2 *(-1)**delta_prime_prime(dx1, dx2)
        return (term1 + C * term2) * sqexp
    
    if derivative_order == 3:                                                                                                       # Returns the third order derivative
        dx1, dx2, dx3 = multi_index_to_x(index)
        term1 = C**3 * check_idx(x1,x2,dx1) * check_idx(x1,x2,dx2) * check_idx(x1,x2,dx3) 
        term2 = (2*(-1)**delta_prime_prime(dx2, dx3) * delta_prime(dx2, dx3) * check_idx(x1, x2, dx1) 
        +  2*(-1)**delta_prime_prime(dx1, dx3) * delta_prime(dx1, dx3) * check_idx(x1, x2, dx2)
        + 2*(-1)**delta_prime_prime(dx1, dx2) * delta_prime(dx1, dx2) * check_idx(x1, x2, dx3))
        return (term1 + C**2 * term2) * sqexp

    if derivative_order == 4:                                                                                                       # Returns the fourth order derivative (MAX.)
        dx1, dx2, dx3, dx4 = multi_index_to_x(index)
        term1 = C**4 * check_idx(x1,x2,dx1) * check_idx(x1,x2,dx2) * check_idx(x1,x2,dx3) * check_idx(x1,x2,dx4)
        term2 = (2*(-1)**delta_prime_prime(dx2, dx3) * delta_prime(dx2, dx3) * check_idx(x1,x2,dx1) * check_idx(x1,x2,dx4) 
        + 2*(-1)**delta_prime_prime(dx3, dx4) * delta_prime(dx3, dx4) * check_idx(x1,x2,dx1) * check_idx(x1,x2,dx2) 
        + 2*(-1)**delta_prime_prime(dx2, dx4) * delta_prime(dx2,  dx4) * check_idx(x1,x2,dx1) * check_idx(x1,x2,dx3)
        + 2*(-1)**delta_prime_prime(dx1, dx2) * delta_prime(dx1, dx2) * check_idx(x1,x2,dx3) * check_idx(x1,x2,dx4)
        + 2*(-1)**delta_prime_prime(dx1, dx3) * delta_prime(dx1, dx3) * check_idx(x1,x2,dx2) * check_idx(x1,x2,dx4)
        + 2*(-1)**delta_prime_prime(dx1, dx4) * delta_prime(dx1, dx4) * check_idx(x1,x2,dx2) * check_idx(x1,x2,dx3))
        term3 = (4*(-1)**(delta_prime_prime(dx1,dx2) + delta_prime_prime(dx3,dx4)) * delta_prime(dx1, dx2) * delta_prime(dx3, dx4)
        + 4*(-1)**(delta_prime_prime(dx1,dx4) + delta_prime_prime(dx2,dx3)) * delta_prime(dx1, dx4) * delta_prime(dx2, dx3)
        + 4*(-1)**(delta_prime_prime(dx1,dx3) + delta_prime_prime(dx2,dx4)) * delta_prime(dx1, dx3) * delta_prime(dx2, dx4))
        return (term1 + C**3 * term2 + C**2 * term3) * sqexp


# Torch differentiations function for 2D vectors only

# To check if first variable, x or second variable, y
def check_variable(idx: int):
    if idx <= 1:
        return 0
    if idx > 1:
        return 1

def torchdiff2D(x: np.ndarray, 
                y: np.ndarray,
                kernelexp, 
                xindex: list,
                yindex: list,
                **kwargs):
    x = tensor(x, requires_grad = True)
    y = tensor(y, requires_grad = True)
    kernel = kernelexp(x, y, **kwargs)
    index = [*xindex, *yindex]
    idx = multi_index_to_x(index)
    gradval = kernel
    for i in idx: 
        var = check_variable(i)
        if var == 0:
            gradval = torchgrad(gradval, x, create_graph = True)[0][i]
        if var == 1:
            gradval = torchgrad(gradval, y, create_graph = True)[0][i-2]
    return gradval

# General squared exponential 
def sqexpnumpy(x: np.ndarray, xprime: np.ndarray, sigma: float, theta: float) -> float:
    sqnorm = np.linalg.norm(x - xprime)**2
    exp = np.exp(-0.5 * sqnorm/(theta**2))
    return sigma**2 * exp

# torch kernel
def torchsqexp2D(x: tensor, y: tensor, sigma: float, theta: float) -> tensor:
    sqnorm = torchnorm(x - y)**2
    exp = torchexp(-0.5 * sqnorm/(theta**2))
    return sigma**2 * exp

if __name__ == '__main__':
    print('Running covderivatives2D.py')
    x = np.array([1.0,2.0])
    y = np.array([2.0,3.0])
    sigma = 2.0
    theta = 1.0
    kwargs = {'sigma': sigma,
              'theta': theta} 
    sympykwargs = {'sigmaval': sigma,
                   'thetaval': theta}
    args = ([0,1],
            [0,0])
    print(f'Analytical: {AnalyticalSqExpdiff2D(x,
                                               y,
                                               *args,
                                               **kwargs)}')
    print(f'Torch2D: {torchdiff2D(x,
                                y,
                                torchsqexp2D,
                                *args,
                                **kwargs)}')
    print(f'sympy2D: {sympydiff2D(x,
                                  y,
                                  sqexp2D,
                                  *args,
                                  **sympykwargs)}')

else: 
    print('Importing covderivatives2D.py')