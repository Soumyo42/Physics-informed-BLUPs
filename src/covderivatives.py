# 1D and 2D Covariance functional derivatives using multi-index notation

# imports
import numpy as np
import random 
from sympy import exp, diff, symbols
from torch import sum as torchsum
from torch import exp as torchexp
from torch import tensor as tensor
from torch.autograd import grad as torchgrad

x1, x2, sigma, theta = symbols('x1 x2 c s')                                 # Setting up the symbols
sqexp1D = sigma**2 * exp(-0.5 * ((x1 - x2)**2
                                    /theta**2))                             # Writing the kernel expression
 
vars = [x1, x2]                                                             # creating a variable list            

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

# Sympy differentiation function for 1D vectors only.
def sympydiff1D(x: float, 
                y: float, 
                kernelexp, 
                xindex: float,
                yindex: float, 
                sigmaval = 1.0, 
                thetaval = 1.0) -> float:
    x1, x2, sigma, theta = symbols('x1 x2 c s')                             # Declaring the variables in local scope
    index = [xindex, yindex]
    vars = [x1, x2] 
    if index == [0,0]:
        return kernelexp.evalf(subs = 
                               {x1:x, 
                                x2:y, 
                                sigma: sigmaval, 
                                theta: thetaval})
    order = multi_index_to_x(index)                                         # Finding the individual components to differentiate by     
    diffvar = [vars[o] for o in order]                                      # Selecting the variables to differentiate by   
    ans = diff(kernelexp, *diffvar).evalf(subs = 
                                          {x1:x, 
                                           x2:y, 
                                           sigma: sigmaval, 
                                           theta: thetaval})                # Final evaluation with values
    return ans


# kernel definition in torch
def sqexptorch(x, y,  
               thetaval: float = 1.0, 
               sigmaval: float = 1.0):
    sqdist = torchsum((x-y)**2)
    return sigmaval**2 * torchexp(-0.5 * sqdist / thetaval**2) # noisy RBF kernel

# To check if first variable, x or second variable, y
def check_variable(idx: int):
    if idx <= 1:
        return 0
    if idx > 1:
        return 1

# Torch differentiations function for 1D vectors only
def torchdiff1D(x: float, 
                y: float,
                kernelexp, 
                xindex: float,
                yindex: float,
                **kwargs):
    x = tensor(x, requires_grad = True)
    y = tensor(y, requires_grad = True)
    kernel = kernelexp(x, y, **kwargs)
    index = [xindex, yindex]
    idx = multi_index_to_x(index)
    gradval = kernel
    for i in idx: 
        var = i
        if var == 0:
            gradval = torchgrad(gradval, x, create_graph = True)[0]
        if var == 1:
            gradval = torchgrad(gradval, y, create_graph = True)[0]
    return gradval.item()

# Analytical differentiation for 1D vectors only
# TODO



# Finite differences 1D, h is the step-size.
# TODO                 
def FiniteDiff(x, y, 
               kernel, 
               index, 
               h: float = 0.5): 
    order = multi_index_to_x(index)
    functions = []
    for i in order: 
        var = i                                                                                # Two if-conditions based on which variable (x or y). Two separate definitions of the findiff functions.
        if var == 0:
            def findiff1(x, y, *funcs):                                                                     # finite differences wrt. x and a particular index. 
                ei = np.zeros_like(x)
                j = i                                                                                       # temporary variable j to avoid unusual error. the function definition must always be defined (even if the "if condition" doesn't apply) 
                if i > 1:                                                                                   # setting j such that it is always less than 1. This is for 2D vectors only. 
                    j -= 2
                ei[j] = 1                                                                                       
                dx = ei * h                                                                                 # (infinitesimal) change in x along component i                                                                 
                if len(funcs)>1:                                                                            # in case of composition of functions.
                    f = funcs[0]
                    return (f(x + dx, y, *funcs[1:]) - f(x - dx, y, *funcs[1:]))/2 * h                      # Centered difference formula. Composes multiple functions together.          
                else: 
                    f = funcs[0]
                    return (f(x + dx, y) - f(x - dx, y))/2 * h                                              # The last function is set to be the kernel
            findiff = findiff1                  
        if var == 1: 
            def findiff2(x, y, *funcs):                                                                     # finite differences wrt.  and a particular index. 
                ei = np.zeros_like(y)
                j = i                                                                                       # temporary variable j to avoid unusual error. the function definition must always be defined (even if the "if condition" doesn't apply)
                if i < 2:                                                                                   # setting j such that it is always greater than or equal to 2. This is for 2D vectors only. 
                    j += 2
                ei[j - 2] = 1
                dy = ei * h                                                                                 # (infinitesimal) change in y along component i 
                if len(funcs)>1:                                                                            # in case of composition of functions.
                    f = funcs[0]
                    return (f(x, y + dy, *funcs[1:]) - f(x, y - dy, *funcs[1:]))/2 * h                      # Centered difference formula. Composes multiple functions together.  
                else: 
                    f = funcs[0]
                    return (f(x, y + dy) - f(x, y - dy))/2 * h                                              # The last function is set to be the kernel
            findiff = findiff2
        functions.append(findiff)                                                                           # storing all the functions in a list
    functions.append(kernel)                                                                                # last function is the kernel function        
    evalf = functions[0](x, y, *functions[1:])                                                              # Calling the first function with arguments x, y and the rest of the functions (composition)    
    return evalf

# General squared exponential 
def sqexpnumpy(x: np.ndarray, xprime: np.ndarray, sigma: float, theta: float) -> float:
    sqnorm = np.linalg.norm(x - xprime)**2
    exp = np.exp(-0.5 * sqnorm/(theta**2))
    return sigma**2 * exp


if __name__ == '__main__':
    print('Running covderivatives.py')
    # import random 
    # L = 0
    # U = 10
    # x = np.random.uniform(L, U)
    # y = np.random.uniform(L, U)
    # sigma = 2.0
    # theta = 1.0
    # kwargs = {'sigma': sigma,
    #           'theta': theta} 
    # sympykwargs = {'sigmaval': sigma,
    #                'thetaval': theta}
    # args = (2,
    #         3)
    # print(f'FiniteDiff: {FiniteDiff(x,
    #                                 y,
    #                                 sqexpnumpy,
    #                                 *args)}')
    # print(f'Torch2D: {torchdiff1D(x,
    #                             y,
    #                             torchsqexp2D,
    #                             *args,
    #                             **kwargs)}')
    # print(f'sympy2D: {sympydiff2D(x,
    #                               y,
    #                               sqexp2D,
    #                               *args,
    #                               **sympykwargs)}')
else: 
    print('Importing covderivatives.py')

