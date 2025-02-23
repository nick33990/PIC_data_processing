import numpy as np
from collections.abc import Iterable
from scipy.interpolate import RegularGridInterpolator, CubicSpline
from collections.abc import Iterable
from tqdm import tqdm
import h5py
import os

from .math_utils import F, Fi
from .constants import *
from .base_map import base_map, inplacify
from .file_utils import load_line, find_params

import matplotlib
matplotlib.use(mpl_backend)
import matplotlib.pyplot as plt
from collections.abc import Iterable

__all__ = ['xt_map', 'SG_hf_filter1D']



def SG_hf_filter1D(dt, size, cent, width, order, cut_hf = True):
    dw = (1) / (size * dt)
    wp = np.arange(0, size // 2) * dw 
    F = np.exp(-(0.5 * (wp - cent) / (width)) ** order)
    if cut_hf:
        F[int(cent/dw):] = 1
    F = np.hstack((F[::-1], F))
    if size % 2 != 0:
        F = np.pad(F, ((1, 0)), constant_values = F[-1])

    return F

class xt_map(base_map):
    """
    class to process and analyze spatiotemporal dynamics 
    """
    def __init__(self, A, dx = 1, dt = 1, x_title = 'x, cells', t_title = 't, steps', x_origin = 0, t_origin = 0):
        super().__init__(A = A, dx = dt, dy = dx, x_title = t_title, y_title = x_title, x_origin = t_origin, y_origin = x_origin)

    @staticmethod
    def from_slices(directory, axis, dx = 1, dt = 1, skip = 2, dtype = 'float32', fn2timestep = lambda s : int(s[6:-4])):
        """
        reads spatiotemporal dynamics from
        """
        files = sorted(os.listdir(directory), key = fn2timestep)
        files = files[::skip]
        lines0 = load_line(os.path.join(directory, files[0]), [axis], dtype = dtype)
        Ny, Nt = len(lines0[0]), len(files)
        F = np.zeros((Nt, Ny), dtype = dtype) 
    
        F[0, :] = lines0[0]
        for j, file in tqdm(enumerate(files[1:])):
            lines0 = load_line(os.path.join(directory, file), axes = [axis], dtype = dtype)
            F[j, :] = lines0[0]
        return xt_map(
            A = F.T,
            dx = dx * 1e6, dt = dt * skip * 1e15,
            x_origin = F.T.shape[0] // 2, t_origin = 0,
            x_title = '$x, \mu m$', t_title = '$t, fs$' 
        )


    @staticmethod
    def from_slices_h5(path, field, skip = 1):
        with h5py.File(path, 'r') as f:
            F = np.array(f['data'][field[0]][field[1]])[0].T
            m = h5py.AttributeManager(f['data'])
            return xt_map(F[::skip], dx = skip * m['dx_SI'] * 1e6, dt = m['dt_SI'] * 1e15,\
                x_title = '$x, \mu m$', t_title = '$t, fs$')


    @staticmethod
    def from_slices_legacy(calc_directory, field, skip_step = 10):
        """
        loads spatiotemporal dynamics from directory, 
        where simulation setup (as .param files) and processed slices (as .npy) are stored    
        """
        files = os.listdir(calc_directory)
        params = ['CELL_WIDTH_SI', 'CELL_HEIGHT_SI',\
        'CFL_RATIO', 'SQRT3', 'DELTA_T_SI']
        p = find_params(calc_directory, 'grid', params)
        for param in params[:-1]:
            p['DELTA_T_SI'] = p['DELTA_T_SI'].replace(param, str(p[param]))
        p['DELTA_T_SI'] = skip_step * eval(p['DELTA_T_SI'].replace('SPEED_OF_LIGHT_SI', str(c)))
        A = np.load(os.path.join(calc_directory, 'yt_maps', 'slices_' + field + '.npy'))
        return xt_map(A.T, dx = 1e6 * p['CELL_HEIGHT_SI'], dt = 1e15 * p['DELTA_T_SI'],\
        x_title = '$x, \mu m$', t_title = '$t, fs$')
        

    @staticmethod
    def from_probes_h5(path, field, filter_fn = None, default_kw = {},\
     shape = None, verbose = True):
        """
        reads spatiotemporal dynamics from single .h5, proccessed by file_utils.probes2h5

        filter_fn - callable: coordinates of probes -> indices of probes to peak, spatial axis 
        """
        with h5py.File(path, 'r') as f:
            F = np.array(f['data'][field[0]][field[1]])
            m = h5py.AttributeManager(f['data'])
            xy = f['data']['xy']
            dx, dt = m['dx_SI'], m['dt_SI']
            print(dt)

            if not filter_fn is None:
                leftover, x_ = filter_fn(xy, **default_kw)
                if (isinstance(x_, tuple) or isinstance(x_, list)) and isinstance(x_[1], str):
                    x_, xtitle = x_[0], x_[1]
                else:
                    xtitle = 'x, a.u.'
                
                sort_idx = sorted(leftover, key = lambda idx:x_[idx])
                F = F[:, sort_idx]
                x_ = x_[sort_idx]
                F = F.reshape((len(F), len(x_)))
                F_interp = np.zeros_like(F)
                x_interp = np.linspace(np.min(x_), np.max(x_), len(x_))

                for k in tqdm(range(len(F))) if verbose else range(len(F)):
                    spl = CubicSpline(x_, F[k])
                    F_interp[k] = spl(x_interp)

                F = F_interp.copy()
                dx = x_[1] - x_[0]


            return xt_map(
                A = F.T,\
                dx = dx, dt = 1e15 * dt,\
                t_title = 't, fs', x_title = xtitle,\
                t_origin = 0, x_origin = 0 
            )


    @inplacify
    def xw(self, new_title = None, crop_negative = True):
        """
        computes FFT along temporal axis
        """
        A_xw = F(self.data, axis = 1)
        if crop_negative:
            A_xw = A_xw[:, A_xw.shape[1] // 2:]
        self.dx = 1 / (self.dx * self.data.shape[1])
        self.data = A_xw
        self.x_title = self.x_title if new_title is None else new_title
        self.x_origin = 0 if crop_negative else -self.dx * len(A_xw) // 2  
        return self
        

    @inplacify
    def kt(self, new_title = None, crop_negative = False):
        """
        computes FFT along spatial axis
        """
        A_xw = F(self.data, axis = 0)
        if crop_negative:
            A_xw = A_xw[A_xw.shape[0] // 2:]
        self.dy = 1 / (self.dy * self.data.shape[0])
        self.data = A_xw
        self.y_title = self.y_title if new_title is None else new_title
        self.y_origin = 0 if crop_negative else -len(A_xw) // 2 * self.dy
        return self

    @inplacify
    def kw(self, crop_negative_freq = True, crop_negative_k = False):
        """
        computes FFT along both axes
        """
        self.xw(inplace = True, crop_negative = crop_negative_freq)
        self.kt(inplace = True, crop_negative = crop_negative_k)
        return self

    @inplacify
    def angle_frequency(self, w2k, show_progress = False):
        """
        computes angle-frequency map. It is assumed, that spatial axis is in cartesian units

        w2k - term in dispersion relation. e.g. if dw, dk in SI units, then w2k = 1/c
          if in normilized to laser frequency and wavevector, then w2k = 1
        show_progress - if show progress bar
        """
        self.kw(inplace = True, crop_negative_freq = True, crop_negative_k = False)
        kx = np.linspace(-0.5, 0.5, self.data.shape[0]) * self.data.shape[0] * self.dy
        w = np.linspace(0, 1, self.data.shape[1]) * (self.data.shape[1]) * self.dx
        theta = np.linspace(-np.pi / 2, np.pi / 2, len(kx))
        sin_theta = np.sin(theta) * w2k
        A_angle_w = np.empty_like(self.data)
        r = tqdm(range(len(w))) if show_progress else range(len(w))
        for i in r:
            spl = CubicSpline(kx, self.data[:, i])
            A_angle_w[:, i] = spl(w[i] * sin_theta)
        theta *= 180 / np.pi
        self.data = A_angle_w
        self.y_title = r'$\theta^{\circ}$'
        self.dy = theta[1] - theta[0]
        self.y_origin = -.5 * self.dy * len(theta)
        return self

    @inplacify
    def apply_direction_filter(self, Filter):
        return super().apply_filter(Filter, axis = (0, 1))

    @inplacify
    def propagate_cart(self, z):
        raise NotImplemented

    @inplacify
    def compensate_curvature(self):
        """
        compensates wavefront curvature by time shift of each line

        Parameters
        ----------

        Returns:
        ----------    
        """

        ph_xw = F(self.data, axis = 0)
        A_w = np.sum(np.abs(ph_xw[self.data.shape[1] // 2:]), axis = 1)
        ph_xw = np.angle(ph_xw)
        w0_idx = np.argmax(A_w) 
        delay = ph_xw[self.data.shape[1] // 2 + w0_idx]
        delay = np.unwrap(delay)
    
        delay -= delay[self.data.shape[0] // 2]
        delay = delay / (np.argmax(A_w) * 2 * np.pi / self.data.shape[1])
    
        A_xt_delayed = np.zeros_like(self.data)
        for i in range(self.data.shape[1]):
            A_xt_delayed[:, i] = np.roll(self.data[:, i], int(delay[i]))
        self.data = A_xt_delayed
        return self


    def show_marginal(self, axs, axis, size = 0.25, post_proc = None, **kw):
        """
        adds sum of data along one of axes to the matplotlib axs 

        axis - axis along which to sum (0 - t, 1 - x)
        """
        xlims, ylims = axs.get_xlim(), axs.get_ylim()
        S = np.sum(self.data, axis = axis)
        S = S if post_proc is None else post_proc(S)
        S = (S - np.min(S)) / (np.max(S) - np.min(S))
        S *= size
        if axis == 0: #x
            x = np.linspace(*xlims, self.data.shape[1])
            y = ylims[0] + S * (ylims[1] - ylims[0])
        else:
            x = xlims[0] + S * (xlims[1] - xlims[0])
            y = np.linspace(*ylims, self.data.shape[0])
        axs.plot(x, y, **kw)
        axs.set_xlim(xlims)
        axs.set_ylim(ylims)

