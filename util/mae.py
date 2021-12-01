import numpy as np

def abs_error(x0,x):
    mae = np.sum(np.absolute(x0 - x))
    return mae
