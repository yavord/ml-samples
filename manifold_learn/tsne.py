import numpy as np
import matplotlib.pylab as pylab

def perp(D=np.array([]), b=1.0):
    P = np.exp(-D.copy()*b)
    sumP = sum(P)
    H = np.log(sumP) + b * np.sum(D*P) / sumP
    P = P / sumP
    return(H, P)

def getPval(X=np.array([]), tol=1e-5, perp=50.0):
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    b = np.ones((n, 1))
    logU = np.log(perp)

    for i in range(n):
        bmin = -np.inf
        bmax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_))]
