import numpy as np
import torch
from itertools import product

def cubic_subproblem_simple_solver(g, A, H, l=-5.0, r=5.0, tol=1e-1):
    """
    Solving by brute force. Only for 2- or 3-dimensional space
    g: np.array, gradient
    A: np.array, Hessian
    H: float, Lipschitz constant
    l: float, left endpoint of the hypercube of search
    r: float, right endpoint of the hypercube of search
    tol: float, tolerance of the error for each coordinate
    
    Returns
    y_best: np.array, approximate solution of the Cubic subproblem
    """
    
    # will be too slow for high dims
    assert A.shape[0] <= 3
    
    def f(y):
        return np.dot(g,y) + 1/2 * np.dot(A@y, y) + H/6 * np.linalg.norm(y)**3
    
    y_best = None
    min_value = np.inf
    num = int(np.ceil((r-l) / tol))
    
    for y in product(*map(lambda x : np.linspace(l, r, num), np.zeros_like(g))):
        value = f(y)
        if (value < min_value):
            y_best = y
            min_value = value
    return y_best


def cubic_subproblem_solver(g, A, H, tolerance=1e-8):
    """
    Finding the exact solution
    g: np.array or torch.tensor, gradient
    A: np.array or torch.tensor, Hessian
    H: float, Lipschitz constant
    tolerance: float, tolerance for the error in solution norm
    
    Returns
    y: np.array or torch.tensor, solution of the Cubic subproblem
    """
    
    # assert g.ndim == 1
    # assert A.ndim == 2 and A.shape[0] == A.shape[1]
    # assert np.allclose(A, A.T), f'A and A.T maximum difference is {np.max(np.abs(A - A.T))}'
    # assert H > 0
    
    if type(g) == torch.Tensor:
        lib = torch
    elif type(g) == np.ndarray:
        lib = np
    else:
        raise TypeError("Unsupported type of inputs: must be torch.Tensor or numpy.array")
    
    # convert to the basis where A is diagonazable
    lambdas, U = lib.linalg.eigh(A) # it returns eigenvalues in ascending order, but we want in descending
    """
    if lib is torch:
        U = lib.flip(U, dims=(1,))
    else:
        U = lib.flip(U, axis=1)
    lambdas = lambdas[::-1]
    """
    g_hat = U.T @ g
    
    def f(r):
        # check division by 0
        zero_denom = (lambdas + H*r/2 == 0)
        if any(g_hat[zero_denom]):
            res = - lib.inf
        else:
            res = r**2 - lib.sum(lib.square(g_hat[~zero_denom] / (lambdas[~zero_denom] + H*r/2)))
        return res
    
    left_endpoint = 2 / H * max(-lambdas[0], 0)
    if f(left_endpoint) > 0:
        # degenerate case
        r = left_endpoint
        n = lambdas.size # space dim
        k = 0
        while (k+1 <= n-1 and lambdas[k+1] == lambdas[0]):
            k += 1
        h = lib.zeros(g_hat.size)
        h[k+1:] = - g_hat[k+1:] / (lambdas[k+1:] + H*r/2)
        h[0] = lib.sqrt(r**2 - lib.dot(h,h))
    else:
        # non-degenerate case
        # finding left and right endpoints
        step = 1
        right_endpoint = left_endpoint + step
        while f(right_endpoint) < 0:
            step *= 2
            right_endpoint = left_endpoint + step
        left_endpoint = right_endpoint - step
        
        # finding r: f(r) = 0 via binary search
        while (right_endpoint - left_endpoint > tolerance):
            midpoint = (right_endpoint + left_endpoint) / 2
            if f(midpoint) < 0:
                left_endpoint = midpoint
            else:
                right_endpoint = midpoint
        r = (right_endpoint + left_endpoint) / 2
        h = - g_hat / (lambdas + H*r/2)
    
    # return to old basis
    y = U @ h
    return y