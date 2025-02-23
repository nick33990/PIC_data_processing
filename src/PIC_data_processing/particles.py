import h5py
import numpy as np
from scipy.signal import convolve2d

from .file_utils import read_positions
from .constants import *

import matplotlib
matplotlib.use(mpl_backend)
import matplotlib.pyplot as plt

__all__ = ['CIC', 'Species']

def CIC(r):
    res = 1 - np.abs(r)
    res[np.where(res < 0)] = 0
    return res

class Species:
    def __init__(self, path, species_name, load_r = True, load_p = True, unit_r = 1e6, unit_p = 1 / (me * c)):
        f = h5py.File(path)
        f = f['data']
        f = f[next(iter(f.keys()))]
        self.attr = dict(h5py.AttributeManager(f))


        attr_m = h5py.AttributeManager(f['particles'][species_name]['mass'])

        self.dr = np.array([self.attr['cell_' + x] for x in ['width', 'height', 'depth']])
        self.dr *= self.attr['unit_length'] * unit_r
        self.mass = attr_m['unitSI']
        self.mass_real = attr_m['value'] * self.mass

        self.weight = np.array(f['particles'][species_name]['weighting'])
        if load_p:
            for k, v in h5py.AttributeManager(f['particles'][species_name]['momentum']['x']).items():
                self.attr['mom' + '_' + k] = v
            self.p = np.array([
                np.array(f['particles'][species_name]['momentum']['x']),
                np.array(f['particles'][species_name]['momentum']['y']),
                np.array(f['particles'][species_name]['momentum']['z'])
            ])
            self.p = self.p.astype('float32') * self.attr['mom_unitSI'] / self.weight * unit_p
        if load_r:
            self.r = np.array(read_positions(f, species_name)).astype('float32')
            self.r *= self.dr[:len(self.r)].reshape(-1, 1)
        else:
            self.r = None
      

    def particles2field(self, dr, shape = 'CIC'):
        min_r = [np.min(self.r[i]) for i in range(len(self.r))]
        Nx, Ny = [1 + int((np.max(self.r[i]) - min_r[i]) / dr[i]) for i in range(len(self.r))]

        pos1D = (((self.r[1] - min_r[1]) / dr[1]).astype('int64')) * Nx
        pos1D += (((self.r[0] - min_r[0]) / dr[0]).astype('int64'))


        grid = np.zeros(Ny * Nx)
        grid[pos1D] = self.weight 
        grid = grid.reshape((Ny, Nx))

        Nkx, Nky = int(4 * self.dr[0] / dr[0]), int(4 * self.dr[1] / dr[1])
        xx, yy = np.mgrid[-Nky // 2:Nky // 2, -Nkx // 2:Nkx // 2]
        xx = xx.astype('float32') * dr[0]
        yy = yy.astype('float32') * dr[1]

        if callable(shape):
            cloud = shape(xx / self.dr[0], yy / self.dr[1])
        elif shape == 'CIC':
            cloud = CIC(xx / self.dr[0]) * CIC(yy / self.dr[1])
        else:
            raise NotImplemented

        cloud /= np.sum(cloud)
        grid = convolve2d(grid, cloud)
        return grid / (dr[0] * dr[1] * self.dr[2])


    def energy_eV(self, mc = 1):
        return self.mass_real * c * c / e * (np.sqrt(1 + np.sum((self.p / mc) ** 2, axis = 0)) - 1)


    def rotate(self, angle):
        if not self.r is None:
            self.r[0], self.r[1] = self.r[0] * np.cos(angle) + self.r[1] * np.sin(angle),\
                                    -self.r[0] * np.sin(angle) + self.r[1] * np.cos(angle)

        if not self.p is None:
            self.p[0], self.p[1] = self.p[0] * np.cos(angle) - self.p[1] * np.sin(angle),\
                                    self.p[0] * np.sin(angle) + self.p[1] * np.cos(angle)


    def filter(self, criterion):
        criterion = np.where(criterion)
        self.weight = self.weight[criterion]
        if not self.r is None:
            self.r = self.r[criterion]
        if not self.p is None:
            self.p = self.p[criterion]

    def show(self, fig = None, axs = None, skip = 1, **kw):
        if axs is None:
            fig, axs = plt.subplots(1, 1)
        axs.scatter(self.r[0][::skip], self.r[1][::skip], **kw)
