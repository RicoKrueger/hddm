# Source: https://www.sfu.ca/sasdoc/sashtml/iml/chap11/sect8.htm
import numpy as np

def hessian(fun, x, eta=np.finfo(np.float64).eps):
    n = x.shape[0]
    hess = np.zeros((n, n))
    for i in range(n):
        hi = eta**(1/3) * (1 + np.abs(x[i]))
        ei = np.zeros(n)
        ei[i] = 1
        for j in range(n):
            if i == j:
                f1 = fun(x + 2 * hi * ei)
                f2 = fun(x + hi * ei)
                f3 = fun(x)
                f4 = fun(x - hi * ei)
                f5 = fun(x - 2 * hi * ei)
                hess[i,j] = (-f1 + 16 * f2 - 30 * f3 + 16 * f4 - f5) / (12 * hi**2)
            else:
                hj = eta**(1/3) * (1 + np.abs(x[j]))
                ej = np.zeros(n)
                ej[j] = 1
                f1 = fun(x + hi * ei + hj * ej)
                f2 = fun(x + hi * ei - hj * ej)
                f3 = fun(x - hi * ei + hj * ej)
                f4 = fun(x - hi * ei - hj * ej)
                hess[i,j] = (f1 - f2 - f3 + f4) / (4 * hi * hj)
    return hess
            
