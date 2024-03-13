import numpy as np

def minmaxscale(tmp, scale_=1):
	if np.count_nonzero(tmp) > 0:
		tmp = tmp - np.amin(tmp)
		tmp = tmp / np.amax(tmp)
		tmp *= scale_
		tmp *= 255
	return tmp

def flatten_(tmp):
	out = []
	for t in tmp:
		for t2 in t:
			out.append(t2)
	out = np.array(out)
	return out
