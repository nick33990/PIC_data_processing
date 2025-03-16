from copy import copy
import numpy as np
from scipy.ndimage import rotate

from .math_utils import fft_filter_F, fft2_filter_F
from .constants import *

import matplotlib
matplotlib.use(mpl_backend)

import matplotlib.pyplot as plt
from collections.abc import Iterable
from mpl_toolkits.axes_grid1 import make_axes_locatable


def inplacify(method):
	def wrap(self, *a, **k):
		inplace = k.pop("inplace", False)
		if inplace:
			method(self, *a, **k)
		else:
			return method(copy(self), *a, **k)
	return wrap


class base_map:
	"""
	base class that implements basic functionality to process spatial and spatiotemporal dymamics
	"""
	def __init__(self, A, dx = 1, dy = 1,\
	 x_origin = 0, y_origin = 0, x_title = 'x, cells', y_title = 'y, cells'):
		self.data = A
		self.dx, self.dy = dx, dy
		self.x_origin, self.y_origin = x_origin, y_origin
		self.shape = A.shape
		self.x_title = x_title
		self.y_title = y_title



	@inplacify
	def apply_filter(self, Filter, axis = None):
		"""
		computes result of Fourier-filtering
		"""
		if axis is None or (isinstance(axis, Iterable) and len(axis) == 2):
			self.data = fft2_filter_F(self.data, Filter)
		elif isinstance(axis, int):
			if len(Filter.shape) == 1:
				Filter = Filter.reshape((-1, 1) if axis == 0 else (1, -1))
			self.data = fft_filter_F(self.data, Filter, axis = axis)
		return self
			

	def _fft_switch_axis(self, title):
		"""
		switches title of axis after transition to Fourier space
		"""
		in_fourier_space = '^-1' in title or 'k' in title
		if in_fourier_space:
			return title.replace('k_', '').replace('k', '').replace('^-1', '')
		else:
			return 'k' + title + '^-1'
	
	@inplacify
	def crop_c(self, xc, yc, wx, wy):
		'''
		Crops data with center in (xc, yc) and width and height equal to wx and wy
		all parameters should be specified in same units as self.dx and self.dy
		'''
		self.crop_b(xc - .5 * wx, xc + .5 * wx,\
							yc - .5 * wy, yc + .5 * wy, inplace = True)
		return self
		
	@inplacify
	def crop_b(self, x0, x1, y0, y1):
		'''
		Crops data by boundaries, specified in same units as self.dx and self.dy
		'''
		assert x0 < x1, 'x0 should be less than x1'
		assert y0 < y1, 'y0 should be less than y1'
		i0, i1 = [int((y - self.y_origin) / self.dy) for y in [y0, y1]]
		j0, j1 = [int((x - self.x_origin) / self.dx) for x in [x0, x1]]
		assert (i0 >= 0) and (j0 >= 0) and (i1 < self.data.shape[0]) and (j1 < self.data.shape[1]), 'Invalid position'

		self.data = self.data[self.data.shape[0] - i1:self.data.shape[0] - i0, j0:j1]

		self.x_origin = x0
		self.y_origin = y0
		return self

	def show(self, fig = None, axs = None, show_colorbar = False, log_scale = False, **kw):
		"""
		draws data on matplotlib axes
		"""
		extent = [0 + self.x_origin, self.data.shape[1] * self.dx + self.x_origin,\
				0 + self.y_origin, self.data.shape[0] * self.dy + self.y_origin]

		if axs is None:
			fig, axs = plt.subplots(1, 1)
		if not isinstance(axs, Iterable):
			axs = [axs]
		im = axs[0].imshow(np.log10(self.data) if log_scale else self.data, extent = extent, **kw)
		show_colorbar = show_colorbar or (len(axs) == 2)
		if show_colorbar:
			if len(axs) == 1:
				divider = make_axes_locatable(axs[0])
				cax = divider.append_axes('right', size='5%', pad=0.05)
				fig.colorbar(im, cax=cax, orientation='vertical')
			else:
				fig.colorbar(im, cax = axs[1])
		if not self.x_title is None:	
			axs[0].set_xlabel(self.x_title)
		if not self.y_title is None:
			axs[0].set_ylabel(self.y_title)

	@inplacify
	def abs(self):
		self.data = np.abs(self.data)
		return self


	@inplacify
	def abs_sqr(self):
		self.data = np.real(self.data * self.data.conjugate())
		return self


	def rescale_axis(self, axis, scale, unit = None):
		"""
		rescales axis scale times. Also changes its title
		"""
		if not isinstance(axis, Iterable):
			axis = [axis]
		if not isinstance(scale, Iterable):
			scale = [scale]
		if isinstance(unit, str) or not isinstance(unit, Iterable):
			unit = [unit]
		if len(scale) < len(axis):
			scale = scale * len(axis)
		if len(unit) < len(axis):
			unit = unit * len(axis)
		
		for i in range(len(axis)): 
			if axis[i] == 0:
				self.dx *= scale[i]
				self.x_title = unit[i] if not unit[i] is None else self.x_title + f'x{round(scale[i], 3)}'
				self.x_origin *= scale[i]
			else:
				self.dy *= scale[i]
				self.y_title = unit[i] if not unit[i] is None else self.y_title + f'x{round(scale[i], 3)}'
				self.y_origin *= scale[i]

	def get_axis_range(self, axis):
		if axis == 0:
			return self.x_origin + np.arange(self.data.shape[1]) * self.dx
		else:
			return self.y_origin + np.arange(self.data.shape[0]) * self.dy

	@inplacify
	def rotate(self, angle):
		self.data = rotate(self.data, angle)
		return self

	def center_grid(self):
		"""
		replaces center to middle of array
		"""
		self.x_origin = -.5 * self.data.shape[1] * self.dx
		self.y_origin = -.5 * self.data.shape[0] * self.dy


	######## operators ########
	@staticmethod
	def _check_dim(map1, map2):
		assert map1.dx == map2.dx, 'dx1 != dx2'
		assert map1.dy == map2.dy, 'dy1 != dy2'

		assert map1.x_origin == map2.x_origin, 'x_origin_1 != x_origin_2'
		assert map1.y_origin == map2.y_origin, 'y_origin_1 != y_origin_2'

	def _check_type(self, map2):
		if isinstance(map2, base_map):
			base_map._check_dim(self, map2)
			inc = map2.data
		elif isinstance(map2, np.ndarray) or isinstance(map2, float) or isinstance(map2, int):
			inc = map2
		else:
			raise TypeError("unsupported operand type(s) for +: '{}' and '{}'").format(self.__class__, type(map2))
		return inc

	def __add__(self, map2):
		new = copy(self)
		new.data = self.data + self._check_type(map2)
		return new
	def __sub__(self, map2):
		new = copy(self)
		new.data = self.data - self._check_type(map2)
		return new
	def __mul__(self, map2):
		new = copy(self)
		new.data = self.data * self._check_type(map2)
		return new
	def __truediv__(self, map2):
		new = copy(self)
		new.data = self.data / self._check_type(map2)
		return new
	def __pow__(self, map2):
		new = copy(self)
		new.data = self.data * self._check_type(map2)
		return new
	def __iadd__(self, map2):
		return self + map2
	def __isub__(self, map2):
		return self - map2
	def __imul__(self, map2):
		return self * map2
	def __idiv__(self, map2):
		return self / map2

	def __getitem__(self, i):
		return self.data[i]
	def __setitem__(self, i, val):
		self.data[i] = val
	def __getslice__(self, i, j, step):
		return self.__getitem__(slice(i, j, step))
	def __setslice__(self, i, j, step, val):
		return self.__setitem__(slice(i, j, step), val)