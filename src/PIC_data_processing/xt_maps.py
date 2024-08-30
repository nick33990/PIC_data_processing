import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Iterable
from scipy.interpolate import RegularGridInterpolator, CubicSpline
from collections.abc import Iterable
from tqdm.notebook import tqdm

from .plot_utils import _get_ticks, _round
from .math_utils import F, Fi
from .constants import *


def compensate_curvature(A_xt):
	"""
	compensates wavefront curvature by time shift of each line

	Parameters
	----------
	A_xt - map, to compensate curvature

	Returns:
	----------
	A_xt_delayed - same map with compensated curvature
	"""
	Nt = len(A_xt)
	Nx = len(A_xt[0])
	ph_xw = F(A_xt, axis = 0)
	A_w = np.sum(np.abs(ph_xw[Nt // 2:]), axis = 1)
	ph_xw = np.angle(ph_xw)
	w0_idx = np.argmax(A_w) 
	delay = ph_xw[Nt // 2 + w0_idx]
	delay = np.unwrap(delay)
	
	delay -= delay[Nx // 2]
	delay = delay / (np.argmax(A_w) * 2 * np.pi / Nt)
	
	A_xt_delayed = np.zeros_like(A_xt)
	for i in range(A_xt.shape[1]):
		A_xt_delayed[:, i] = np.roll(A_xt[:, i], int(delay[i]))
	return A_xt_delayed


def plot_xt(fig, axs, A_xt, x, t,\
			xlim = [None, None], tlim = [None, None],\
			num_xticks = 5, num_tticks = 5, round_x = 3, round_t = 3, axis_sum = None, sum_height = 0.2,\
			log_scale = False, sqr_sum = False,\
			imshow_kwargs = {}, sum_kwargs = {'c':'w'}):
	"""
	plots xt map with sum of it along one axis

	Parameters
	----------
	fig - matplotlib figure
	axs - matplotlib axes. if axs is list of two axes objects, than first is used to plot map
		and second - for colorbar
	A_xt - map to plot
	x - list of coordinates along spacial axis ( in plot will be vertical)
	t - list of coordinates along temporal axis ( in plot will be horizontal)
	xlim - borders for spacial axis
	tlim - borders for temporal axis
	num_xticks - number of ticks to plot scacial axis
	num_tticks - number of ticks to plot temporal axis
	round_ - number of significant digits for plot
	axis_sum - axis, along which map needs to be summed
	sum_height - part of figure to draw sum
	log_scale - plot xt map and sum in log scale 
	sqr_sum - sum squared array or raw values
	imshow_kwargs - additional arguments for plotting xt map
	sum_kwargs - additional arguments for plottinf sum
	"""
	assert A_xt.shape[0] == len(t), 'A_xt.shape[0] ({}) != t.shape ({})'.format(A_xt.shape[0], len(t))
	assert A_xt.shape[1] == len(x), 'A_xt.shape[1] ({}) != x.shape ({})'.format(A_xt.shape[1], len(x))

    
	if not isinstance(axs, Iterable):
		axs = [axs]

	xticks = _get_ticks(x, xlim, num_xticks)
	tticks = _get_ticks(t, tlim, num_tticks)
	A_xt = A_xt[tticks[0]:tticks[-1], xticks[0]:xticks[-1]]
	

	im = axs[0].imshow(A_xt.T if not log_scale else np.log10(A_xt.T),\
					   aspect = 'auto', **imshow_kwargs)
	
	axs[0].set_xticks(tticks - tticks[0])
	axs[0].set_xticklabels(_round(t[tticks], round_t))
	axs[0].set_yticks(xticks - xticks[0])

	axs[0].set_yticklabels(_round(x[xticks], round_x))
	if len(axs) == 2:
		fig.colorbar(im, cax = axs[1])
	
	if not isinstance(axis_sum, Iterable):
		axis_sum = [axis_sum]
	if not isinstance(sqr_sum, Iterable):
		sqr_sum = [sqr_sum]
	assert len(axis_sum) == len(sqr_sum), 'lenght of axis_sum should be equal to lenght of sqr_sum'
	for i, axis_sum_ in enumerate(axis_sum):
		if not axis_sum_ is None:
			s = np.sum(A_xt if not sqr_sum[i] else A_xt ** 2, axis = axis_sum_)
			s = s if not log_scale else np.log10(s)
			min_, max_ = np.min(s), np.max(s)
			s = (s - min_) / (max_ - min_)
		
			if axis_sum_ == 1:
				s = A_xt.shape[axis_sum_] * (1 - s * sum_height)
				axs[0].plot(s, **sum_kwargs)
				axs[0].set_ylim([0, xticks[-1] - xticks[0]])
# 				axs[0].axhline(0, c = 'r')
# 				axs[0].axhline(xticks[-1] - xticks[0])
			else:
				s = A_xt.shape[axis_sum_] * s * sum_height	
				axs[0].plot(s, np.arange(len(s)), **sum_kwargs)
				axs[0].set_xlim([0, tticks[-1] - tticks[0]])

    
            
	if len(axis_sum) == 1 and not axis_sum[0] is None:
		axs[0].invert_yaxis()



def compress_xt(F, tr_time = 1e-5, tr_space = 1e-5):
	"""
	compresses xt map, leaving only part, where time averaged values values > tr_time * max(F)
	and space averaged > tr_space * max(F)

	Parameters
	----------
	F - xt map to compress
	tr_time - threshold value, to truncate time axis
	tr_space - threshold value, to truncate space axis

	Returns:
	----------
	cropped xt map
	"""
	mean_B = np.mean(np.abs(F), axis = 1)
	start = np.where(mean_B > tr_time * np.max(mean_B))[0][0]
	F = F[start:]
	mean_B = np.mean(np.abs(F), axis = 0)
	nz = np.where(mean_B > tr_space * np.max(mean_B))[0]
	return F[:, nz[0]: nz[-1]].astype('float32')



def kw2angle_w(ky, w, A_kw, show_progress = False):
	"""
	converts wavevector y-projection frequecny map (A(ky,w)) to
	angle frequency map A(theta,w) using interpolation: A(theta, w) = A(ky=w/c sin theta, w)

	Parameters
	----------
	ky - wavevector projection values (in SI units)
	w - frequency values (in SI units)
	A_kw - wavevector y-projection frequecny map
	show_progress - display progress bar or not

	Returns:
	----------
	angle values, frequency values and angle frequency map
	"""
	theta = np.linspace(-np.pi/2, np.pi/2, len(ky))
	sin_theta = np.sin(theta) / c
	A_angle_w = np.empty_like(A_kw)
	r = tqdm(range(A_kw.shape[0])) if show_progress else range(A_kw.shape[0])
	for i in r:
		spl = CubicSpline(ky, A_kw[i])
 
		A_angle_w[i] = spl(w[i] * sin_theta)
	return theta, w, A_angle_w











# def plot_spectra(f)
# def compensate_diffraction(A_xt):
# 	"""
# 	compensates 
# 	A_xt - map to compensate
# 	"""
#	 Nt = len(A_xt)
#	 Nx = len(A_xt[0])
#	 A_w = np.zeros(Nt // 2)
#	 ph_xw = np.zeros_like(A_xt)
#	 for i in range(A_xt.shape[1]):
#		 tmp = F(A_xt[:, i])
#		 A_w += np.abs(tmp[Nt // 2:])
#		 ph_xw[:, i] = np.angle(tmp)
# #		 ph_xw[:, i] = np.imag(np.arccosh(tmp))
#	 w0_idx = np.argmax(A_w) 
#	 delay = ph_xw[Nt // 2 + w0_idx]
#	 delay = np.unwrap(delay)
	
#	 delay -= delay[Nx // 2]
#	 delay = delay / (np.argmax(A_w) * 2 * np.pi / Nt)
	
#	 A_xt_delayed = np.zeros_like(A_xt)
#	 for i in range(A_xt.shape[1]):
#		 A_xt_delayed[:, i] = np.roll(A_xt[:, i], int(delay[i]))
#	 return A_xt_delayed