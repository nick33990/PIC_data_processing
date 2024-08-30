import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def _round(X, n = 1):
	# r = -(np.floor(np.log10(np.abs(X)))).astype('int32') + (n - 1)
	# return np.round(X, np.min(r[r != -2147483646]))
	r = [-(np.floor(np.log10(np.abs(x)))) + (n - 1) if x != 0 else np.inf for x in X]
	return np.round(X, int(np.min(r[r != np.inf])))

def _get_ticks(x, lim, num_ticks):
	lim = [np.min(x), np.max(x)] if (lim is None or lim[0] is None) else lim
	ticks = np.linspace(lim[0], lim[1], num_ticks)
	ticks = [np.abs(ticks[i] - x).argmin() for i in range(len(ticks))]
	return ticks

def two_color(c0, c1, γ = 1):
	newcolors = np.zeros((256, 3))
#	 c0, c1 = np.array([0, 0, 0]), np.array([1, 0, 0])
	t = np.linspace(0, 1, len(newcolors))[:, None]
	newcolors[:len(t)] = c0 + t ** γ * (c1 - c0)
	return ListedColormap(newcolors)

def three_color(c0, c1, cmid, γ = 1):
	newcolors = np.zeros((256, 3))
	t = np.linspace(0, 1, len(newcolors) // 2)[:, None]
	newcolors[:len(t)] = c0 + t ** γ * (cmid - c0)
	newcolors[len(t):] = cmid + t ** (1 / γ) * (c1 - cmid)
	return ListedColormap(newcolors)


def with_white(orig_cmap, white_center = 127, white_width = 4):
    N = 256
    t = np.linspace(0, 1, N)
    newcolors = orig_cmap(t)[:, :-1]
    start, stop = white_center - white_width, white_center + white_width 
    w = slice(max(0, start + 1),\
              min(N, stop), 1)
    t1 = np.arange(-white_width, white_width + 1)
    gamma = (t[w] - t[(start + stop) // 2]) ** 2# ** 2 / (t[w.start] - t[0] + 1e-6) ** 2
    gamma /= (1e-6 + np.max(gamma))
    gamma = gamma[:, None]
    newcolors[w] = (1 - gamma) * np.array([1, 1, 1]) + gamma * newcolors[w]
    return ListedColormap(newcolors)




	# N = 256
	# t = np.linspace(0, 1, N)
	# newcolors = orig_cmap(t)[:, :-1]
	# w = slice(white_center - white_width, white_center + white_width, 1)
	# γ = (t[w] - 0.5) ** 2 / (t[w.start] - 0.5) ** 2
	# γ = γ[:, None]
	# newcolors[w] = (1 - γ) * np.array([1, 1, 1]) + γ * newcolors[w] 
	# if inv:
	# 	newcolors = newcolors[::-1]
	# return ListedColormap(newcolors)