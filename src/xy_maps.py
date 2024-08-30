import numpy as np
import matplotlib.pyplot as plt
from PIC_lib.plot_utils import _get_ticks, _round

def plot_xy(fig, axs, A, x, y, num_xticks = 5, num_yticks = 5,\
 		round_ = 3, xlim = [None, None], ylim = [None, None],\
 	 	imshow_kwargs = {}):
	xticks = _get_ticks(x, xlim, num_xticks)
	yticks = _get_ticks(y, ylim, num_yticks)
	
	A = A[yticks[0]:yticks[-1], xticks[0]:xticks[-1]]
	im = axs[0].imshow(A, aspect = 'auto', **imshow_kwargs)

	axs[0].set_xticks(xticks - xticks[0])
	axs[0].set_xticklabels(_round(x[xticks], round_))
	axs[0].set_yticks(yticks - yticks[0])
	axs[0].set_yticklabels(_round(y[yticks], round_))
	# axs[0].set_xlim(xlim)
	# axs[0].set_ylim(ylim)
	# axs[0].set_xlim([0, xticks[-1] - xticks[0]])
	# axs[0].set_ylim([0, yticks[-1] - yticks[0]])
	fig.colorbar(im, cax = axs[1])


def plot_overlap_maps(fig, axs, A1, A2, x, y, num_xticks = 5, num_yticks = 5,\
 		round_ = 3, xlim = [None, None], ylim = [None, None],\
 	 	imshow1_kwargs = {}, imshow2_kwargs = {}):
	im1 = axs[0].imshow(A1, aspect = 'auto', **imshow1_kwargs)
	im2 = axs[0].imshow(A2, aspect = 'auto', **imshow2_kwargs)
	xticks = _get_ticks(x, xlim, num_xticks)
	yticks = _get_ticks(y, ylim, num_yticks)
	axs[0].set_xticks(xticks)
	axs[0].set_xticklabels(_round(x[xticks], round_))
	axs[0].set_yticks(yticks)
	axs[0].set_yticklabels(_round(y[yticks], round_))
	axs[0].set_xlim(xlim)
	axs[0].set_ylim(ylim)
	fig.colorbar(im1, cax = axs[1])
	fig.colorbar(im2, cax = axs[2])
