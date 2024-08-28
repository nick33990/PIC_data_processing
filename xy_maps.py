import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Iterable
from .plot_utils import _get_ticks, _round

def plot_xy(fig, axs, A, x, y, num_xticks = 5, num_yticks = 5,\
 		round_x = 3, round_y = 3, xlim = [None, None], ylim = [None, None],\
 	 	imshow_kwargs = {}):
	"""
	plots 2D array on regular grid

	Parameters
	----------
	fig - matplotlib figure object
	axs - matplotlib axes. if axs is list of two axes objects, than first is used to plot map
		and second - for colorbar
	A - 2D array to be plot
	x - array of x-axis values
	y - array of y-axis values
	num_xticks - number of ticks on x-axis
	num_yticks - number of ticks on y-axis
	round_x - number of significant digits to round x-axis values
	round_y - number of significant digits to round y-axis values
	xlim - limits for x-axis
	ylim - limits for y-axis
	imshow_kwargs - additional parameters for imshow
	"""
	if not isinstance(axs, Iterable):
		axs = [axs]

	xticks = _get_ticks(x, xlim, num_xticks)
	yticks = _get_ticks(y, ylim, num_yticks)
	
	A = A[yticks[0]:yticks[-1], xticks[0]:xticks[-1]]
	im = axs[0].imshow(A, aspect = 'auto', **imshow_kwargs)

	axs[0].set_xticks(xticks - xticks[0])
	axs[0].set_xticklabels(_round(x[xticks], round_x))
	axs[0].set_yticks(yticks - yticks[0])
	axs[0].set_yticklabels(_round(y[yticks], round_y))

	if len(axs) == 2:
		fig.colorbar(im, cax = axs[1])


def plot_overlap_maps(fig, axs, A1, A2, x, y, num_xticks = 5, num_yticks = 5,\
 		round_x = 3, round_y = 3, xlim = [None, None], ylim = [None, None],  alpha1 = 1, alpha2 = 1,\
 	 	imshow1_kwargs = {}, imshow2_kwargs = {}):
	"""
	plots two 2D arrays on regular grid using same matplotlib axes. It is recomended to use
	add_transparecy function from .plot_utils to create colormap for second array

	Parameters
	----------
	fig - matplotlib figure object
	axs - matplotlib axes. if axs is list of three axes objects, than first is used to plot map
		, second and third - for colorbars or A1 and A2 arrays
	A1 - first 2D array to be plot
	A2 - first 2D array to be plot
	x - array of x-axis values
	y - array of y-axis values
	num_xticks - number of ticks on x-axis
	num_yticks - number of ticks on y-axis
	round_x - number of significant digits to round x-axis values
	round_y - number of significant digits to round y-axis values
	xlim - limits for x-axis
	ylim - limits for y-axis
	alpha1 - transparency for plotting first array
	alpha2 - transparency for plotting second array
	imshow1_kwargs - additional parameters for imshow of A1 array
	imshow2_kwargs - additional parameters for imshow of A2 array
	"""
	if not isinstance(axs, Iterable):
		axs = [axs]
	
	xticks = _get_ticks(x, xlim, num_xticks)
	yticks = _get_ticks(y, ylim, num_yticks)


	im1 = axs[0].imshow(A1[yticks[0]:yticks[-1], xticks[0]:xticks[-1]],\
	 aspect = 'auto', alpha = alpha1, **imshow1_kwargs)
	im2 = axs[0].imshow(A2[yticks[0]:yticks[-1], xticks[0]:xticks[-1]],\
	 aspect = 'auto', alpha = alpha2, **imshow2_kwargs)


	axs[0].set_xticks(xticks - xticks[0])
	axs[0].set_xticklabels(_round(x[xticks], round_x))
	axs[0].set_yticks(yticks - yticks[0])
	axs[0].set_yticklabels(_round(y[yticks], round_y))

	if len(axs) == 3:
		fig.colorbar(im1, cax = axs[1])
		fig.colorbar(im2, cax = axs[2])
