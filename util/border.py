import numpy as np

def create_border_array(n, m):
	array = np.zeros(shape=(n-1,m-1)) # array of zeroes without border
	array = np.pad(array=array, pad_width=1, mode="constant",constant_values=1) # add border of 1's
	return(array)