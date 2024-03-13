import numpy as np

def minmaxscale(tmp, scale_=1):
	if np.count_nonzero(tmp) > 0:
		tmp = tmp - np.amin(tmp)
		tmp = tmp / np.amax(tmp)
		tmp *= scale_
		tmp *= 255
	return tmp