import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

"""
Parameters
----------
Returns
----------
"""

def _getlims(lim, step):
	"""
	computes indices where from limits
	"""
	lower = int(lim[0] / step) if not lim[0] is None else 0
	upper = int(lim[1] / step) + 1 + 1 if not lim[1] is None else -1
	return slice(lower, upper, 1)

def _round(X, n = 1):
	"""
	rounds all values in X to n significant digits,
	 e.g [0.12e-12, 1.45e-12, 4.56e-12] -> [1.0e-13 1.4e-12 4.6e-12],

	Parameters
	----------
	X - array to be rounded
	n - number of signinficant digits to keep
	Returns
	----------
	rounded array
	"""
	r = [-(np.floor(np.log10(np.abs(x)))) + (n - 1) if x != 0 else np.inf for x in X]
	return np.round(X, int(np.min(r[r != np.inf])))

def _get_ticks(x, lim, num_ticks):
	lim = [np.min(x) if (lim is None or lim[0] is None) else lim[0],
			np.max(x) if (lim is None or lim[1] is None) else lim[1]]
	ticks = np.linspace(lim[0], lim[1], num_ticks)
	ticks = [np.abs(ticks[i] - x).argmin() for i in range(len(ticks))]
	return ticks

def two_color(c0, c1, gamma = 1):
	"""
	creates colormap from two colors, where minimum value correspond to c0
	and maximum to c1

	Parameters
	----------
	c0, c1 - first and last color
	gamma - power, indicates how fast rgb components varies from c0 to c1
	
	Returns
	----------
	Colormap
	"""
	newcolors = np.zeros((256, 3))
	t = np.linspace(0, 1, len(newcolors))[:, None]
	newcolors[:len(t)] = c0 + t ** gamma * (c1 - c0)
	return ListedColormap(newcolors)

def three_color(c0, c1, cmid, gamma = 1):
	"""
	creates colormap from two colors, with one color between them, where minimum value correspond to c0
	medium to cmid and maximum to c1

	Parameters
	----------
	c0, cmid, c1 - first, middle and last color
	gamma - power, indicates how fast rgb components varies from c0 to c1
	
	Returns
	----------
	Colormap
	"""
	newcolors = np.zeros((256, 3))
	t = np.linspace(0, 1, len(newcolors) // 2)[:, None]
	newcolors[:len(t)] = c0 + t ** gamma * (cmid - c0)
	newcolors[len(t):] = cmid + t ** (1 / gamma) * (c1 - cmid)
	return ListedColormap(newcolors)


def with_white(orig_cmap, white_center = 127, white_width = 4):
	"""
	adds white part to some values of existing colormap

	Parameters
	----------
	orig_cmap - colormap, to add white to
	white_center - position to add white part
	white_width - width of white part
	
	Returns
	----------
	Colormap
	"""
	N = 256
	t = np.linspace(0, 1, N)
	newcolors = orig_cmap(t)[:, :-1]
	start, stop = white_center - white_width, white_center + white_width 
	w = slice(max(0, start + 1),\
			  min(N, stop), 1)
	t1 = np.arange(-white_width, white_width + 1)
	gamma = (t[w] - t[(start + stop) // 2]) ** 2
	gamma /= (1e-6 + np.max(gamma))
	gamma = gamma[:, None]
	newcolors[w] = (1 - gamma) * np.array([1, 1, 1]) + gamma * newcolors[w]
	return ListedColormap(newcolors)


def add_transparency(orig_cmap, transparency_func):
	"""
	adds transparency to part of colormap
	Parameters
	----------
	orig_cmap - colormap to process
	transparency_func - function, that returns alpha component of color

	Returns
	----------
	Processed colormap
	"""
	N = 256
	t = np.linspace(0, 1, N)
	colors = orig_cmap(t)
	colors[:, 3] = transparency_func(t)
	return ListedColormap(colors)