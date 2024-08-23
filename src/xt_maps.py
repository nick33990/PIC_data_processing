import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Iterable
from scipy.interpolate import RegularGridInterpolator

from PIC_lib.plot_utils import _get_ticks, _round


def compensate_curvature(A_xt):
	"""
	compensates wavefront curvature by time shift for each line
	A_xt - map, to compensate curvature
	Returns:
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
			num_xticks = 5, num_tticks = 5, round_ = 3, axis_sum = None, sum_height = 0.2,\
			log_scale = False,\
			imshow_kwargs = {}, sum_kwargs = {'c':'w'}):
	"""
	
	"""
	xticks = _get_ticks(x, xlim, num_xticks)
	tticks = _get_ticks(t, tlim, num_tticks)
	A_xt = A_xt[tticks[0]:tticks[-1], xticks[0]:xticks[-1]]
	
	im = axs[0].imshow(A_xt.T if not log_scale else np.log10(A_xt.T),\
					   aspect = 'auto', **imshow_kwargs)
	
	axs[0].set_xticks(tticks - tticks[0])
	axs[0].set_xticklabels(_round(t[tticks], round_))
	axs[0].set_yticks(xticks - xticks[0])
	if xticks[0] < xticks[-1]:
		xticks = xticks[::-1]

	axs[0].set_yticklabels(_round(x[xticks], round_))
	fig.colorbar(im, cax = axs[1])
	
	if not axis_sum is None:
		s = np.sum(A_xt, axis = axis_sum)
		s = s if not log_scale else np.log10(s)
		min_, max_ = np.min(s), np.max(s)
		s = (s - min_) / (max_ - min_)
		s = A_xt.shape[axis_sum] * (1 - s * sum_height)
		axs[0].plot(s, **sum_kwargs)
		axs[0].set_ylim([0, xticks[0] - xticks[-1]])
		axs[0].invert_yaxis()


def cart2pol_interp(x, y, A, squeeze = True):
    """
    transforms array on cartesian grid A(x, y) to polar grid A(r, theta).
    x, y - arrays of coordinates
    A - array to be transformed
    Returns:
    r, theta - arrays or polar coordinates r = sqrt(x^2 + y^2) and angles (in degrees),
    A_ra - array in polar coordinates
    """
    Nx, Ny = A.shape

    borders_x = np.min(x), np.max(x)
    borders_y = np.min(y), np.max(y)
    interp = RegularGridInterpolator((x, y), A, bounds_error = False, fill_value = 0)
    kr, theta = np.linspace(0, np.sqrt(x[-1] ** 2 + y[-1] ** 2), len(x)),\
                np.linspace(0, 2 * np.pi, len(x))
    kr, theta = np.meshgrid(kr, theta)
    Kx, Ky = kr * np.cos(theta), kr * np.sin(theta)
    coords = np.vstack((Kx.flatten(), Ky.flatten())).T
    theta *= 180 / np.pi
    return kr[0], theta[:, 0], interp(coords).reshape(Kx.shape)



















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