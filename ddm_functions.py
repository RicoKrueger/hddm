import numpy as np
from numba import njit

@njit
def ddm_lpdf(y, mu, w, a):
    """ Calculates log of defective PDF of DDM """   
    u = y / a**2    

    #Calculate required number of terms to approximate infinite sum for large t
    err = 1e-10
    bl = 1 / np.pi / np.sqrt(u)
    if (np.pi * u * err) < 1:
        kl = np.sqrt(-2 * np.log(np.pi * u * err) / (np.pi**2 * u))
        kl = np.max(np.array([kl, bl]))
    else:
        kl = bl

    kl_upper = int(np.ceil(kl))
    f = 0
    for k in np.arange(1, kl_upper + 1):
        in_sin_k = np.pi * k * w
        in_exp_k = -0.5 * k**2 * np.pi**2 * u
        f += k * np.sin(in_sin_k) * np.exp(in_exp_k)
    if f < 1e-10:
        #print('f below zero.', f)
        f = 1e-10
    lf = np.log(f) + np.log(np.pi)

    lpdf = lf - 2 * np.log(a)
    lpdf += (-mu * a * w - mu**2 * y / 2)
    return lpdf

@njit
def ddm_p(mu, w, a):
    in_exp = -2 * mu * a * (1 - w)
    if in_exp > 1e2:
        return 1
    elif np.abs(mu) < 1e-5 or w == 1:
        return 1 - w
    else:
        e = np.exp(in_exp)
        return (1 - e) / (np.exp(2 * mu * a * w) - e)

@njit
def ddm_lccdf(y, mu, w, a, return_lccdf=True):
    """ Calculates defective CCDF and CDF of DDM """
    #Calculate required number of terms to approximate infinite sum for large t
    err = 1e-10
    kl1 = y**(-2) * a / np.pi
    in_log = err * np.pi * y / 2 * (mu**2 + np.pi**2 / a**2)
    kl2 = np.sqrt(-2 / y * a**2 / np.pi**2 * (np.log(in_log) + mu * a * w + mu**2 * y / 2))
    kl2 = np.max(np.array([kl2, 1.0]))
    kl = np.max(np.array([kl1, kl2]))

    kl_upper = int(np.ceil(kl))
    F = 0
    in_exp = -mu * a * w - mu**2 * y / 2
    for k in np.arange(1, kl_upper+1):
        in_sin_k = np.pi * k * w
        in_exp_k = -0.5 * (k * np.pi / a)**2 * y + in_exp
        F_k = k * np.sin(in_sin_k) * np.exp(in_exp_k)
        F_k /= (mu**2 + (k * np.pi / a)**2)
        F += F_k
    F *= (-2 * np.pi / a**2)

    P = ddm_p(mu, w, a)
    cdf = P + F
    if return_lccdf:
        ccdf = 1 - cdf
        lccdf = np.log(ccdf)
        return lccdf
    else:
        return cdf

@njit
def ddm_lp(y, mu, w, a, nt, n_ind, n_draws, n_sit):
    """ Calculates log probability of DDM """
    lp = np.zeros((n_ind, n_draws, n_sit))
    for i in range(n_ind):
        for j in range(n_draws):
            for l in range(n_sit):
                if np.isnan(y[i,l]):
                    lp[i,j,l] = ddm_lccdf(15 - nt[i,j,l], -mu[i,j,l], 1 - w[i,j,l], a[i,j,l])
                else:
                    lp[i,j,l] = ddm_lpdf(y[i,l] - nt[i,j,l], -mu[i,j,l], 1 - w[i,j,l], a[i,j,l])
    return lp

@njit
def ddm_cdf(y, mu, w, a, nt, n_ind, n_draws, n_sit, t_steps):
    """ Calculates CDF of DDM at different time steps """
    cdf = np.zeros((n_ind, n_draws, n_sit, t_steps.shape[0]))
    for i in range(n_ind):
        for j in range(n_draws):
            for l in range(n_sit):
                for s, t in enumerate(t_steps):
                    if np.isinf(t):
                        cdf[i,j,l,s] = ddm_p(-mu[i,j,l], 1 - w[i,j,l], a[i,j,l])
                    else:
                        cdf[i,j,l,s] = ddm_lccdf(t - nt[i,j,l], -mu[i,j,l], 1 - w[i,j,l], a[i,j,l], False)
    return cdf

@njit
def coth(x):
    e2x = np.exp(2 * x)
    return (e2x + 1) / (e2x - 1)

@njit
def ddm_rt_upper(mu, w, a, nt, n_ind, n_draws, n_sit):
    """ Calculates expected response time for absorption at upper boundary """
    rt = np.zeros((n_ind, n_draws, n_sit))
    for i in range(n_ind):
        for j in range(n_draws):
            for l in range(n_sit):
                B = a[i,j,l] * w[i,j,l]
                if mu[i,j,l] == 0:
                    A = a[i,j,l] * (1 - w[i,j,l])
                    rt[i,j,l] = (A**2 + 2 * A * B) / 3
                else:
                    rt[i,j,l] = a[i,j,l] / mu[i,j,l] * coth(a[i,j,l] * mu[i,j,l]) - B / mu[i,j,l] * coth(B * mu[i,j,l])
    return rt

@njit
def ddm_draw(mu, w, a, nt):
    t_end = 15 
    delta_t = 0.01
    sqrt_delta_t = np.sqrt(delta_t)
    t = nt
    z = w * a
    
    while True:
        t += delta_t
        if t >= t_end:
            return np.nan
        eta = np.random.randn()
        z += mu * delta_t + sqrt_delta_t * eta
        if z >= a:
            return t
        if z <= 0:
            return np.nan

@njit
def ddm_gen(mu, w, a, nt, n_ind, n_sit):
    """ Generates decisions under DDM """
    y = np.zeros((n_ind, n_sit))
    j = 0
    for i in range(n_ind):
        for l in range(n_sit):
            y[i,l] = ddm_draw(mu[i,j,l], w[i,j,l], a[i,j,l], nt[i,j,l])
    return y