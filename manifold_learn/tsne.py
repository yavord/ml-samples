import numpy as np

def perp(D=np.array([]), b=1.0):
    P = np.exp(-D.copy()*b)
    sumP = sum(P)
    H = np.log(sumP) + b * np.sum(D*P) / sumP
    P = P / sumP
    return(H, P)

def getPval(X=np.array([]), tol=1e-5, perp=50.0):
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)

