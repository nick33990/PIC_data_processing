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
    def __init__(self, weight = None, r = None, p = None, ids = None, dr = None, mass = 1, attr = None):
        self.weight = weight
        self.r = r
        self.p = p
        self.id = ids
        self.dr = dr
        self.mass = mass
        self.attr = attr

    @staticmethod
    def from_openPMD(path, species_name, load_r = True, load_p = True, load_id = True,\
    unit_r = 1e6, unit_p = 1 / (me * c)):
        f = h5py.File(path)
        f = f['data']
        f = f[next(iter(f.keys()))]
        attr = dict(h5py.AttributeManager(f))


        attr_m = h5py.AttributeManager(f['particles'][species_name]['mass'])

        dr = np.array([attr['cell_' + x] for x in ['width', 'height', 'depth']])
        dr *= attr['unit_length'] / unit_r
        mass = attr_m['unitSI']
        mass = attr_m['value'] * mass

        weight = np.array(f['particles'][species_name]['weighting'])
        ids = np.array(f['particles'][species_name]['id']) if load_id else None
        if load_p:
            for k, v in h5py.AttributeManager(f['particles'][species_name]['momentum']['x']).items():
                attr['mom' + '_' + k] = v
            p = np.array([
                np.array(f['particles'][species_name]['momentum'][i]) for i in ['x', 'y', 'z']
            ])
            p = p.astype('float32') * attr['mom_unitSI'] / weight * unit_p
        else:
            p = None
            
        if load_r:
            r = np.array(read_positions(f, species_name)).astype('float32')
            r *= dr[:len(r)].reshape(-1, 1)
        else:
            r = None

        return Species(weight, r, p, ids, dr, mass, attr)

    def particles2field(self, dr, shape = 'CIC', field = 1, as_density = True):
        min_r = [np.min(self.r[i]) for i in range(len(self.r))]
        Nx, Ny = [1 + int((np.max(self.r[i]) - min_r[i]) / dr[i]) for i in range(len(self.r))]

        pos1D = (((self.r[1] - min_r[1]) / dr[1]).astype('int64')) * Nx
        pos1D += (((self.r[0] - min_r[0]) / dr[0]).astype('int64'))


        grid = np.zeros(Ny * Nx)
        field *= self.weight
        for i in range(len(pos1D)):
            grid[pos1D[i]] += field[i]
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
        if as_density:
            grid /= (dr[0] * dr[1] * self.dr[2])

        extent = [min_r[0],\
                  min_r[0] + grid.shape[1] * self.dr[0],\
                  min_r[1],
                  min_r[1] + grid.shape[0] * self.dr[1]]
        return grid, extent


    def sort_by_id(self):
        idx = np.argsort(self.id)
        self.id = self.id[idx]
        self.weight = self.weight[idx]
        if not self.r is None:
            self.r[:, ] = self.r[:, idx]
        if not self.p is None:
            self.p[:, ] = self.p[:, idx]

    def remove_ids(self, to_remove):
        idx = np.arange(len(self.id))
        to_remove = idx[np.searchsorted(self.id, to_remove, sorter = idx)]

        self.id = np.delete(self.id, to_remove)
        self.weight = np.delete(self.weight, to_remove)
        if not self.r is None:
            self.r = np.delete(self.r, to_remove, 1)
        if not self.p is None:
            self.p = np.delete(self.p, to_remove, 1)

    def energy_eV(self, mc = 1):
        return self.mass * c * c / e * (np.sqrt(1 + np.sum((self.p / mc) ** 2, axis = 0)) - 1)


    def rotate(self, angle):
        if not self.r is None:
            self.r[0], self.r[1] = self.r[0] * np.cos(angle) + self.r[1] * np.sin(angle),\
                                    -self.r[0] * np.sin(angle) + self.r[1] * np.cos(angle)

        if not self.p is None:
            self.p[0], self.p[1] = self.p[0] * np.cos(angle) + self.p[1] * np.sin(angle),\
                                    -self.p[0] * np.sin(angle) + self.p[1] * np.cos(angle)


    def filter(self, criterion):
        criterion = np.where(criterion)
        self.weight = self.weight[criterion]
        if not self.r is None:
            self.r = self.r[:, criterion].reshape(-1, len(self.weight))
        if not self.p is None:
            self.p = self.p[:, criterion].reshape(-1, len(self.weight))
        if not self.id is None:
            self.id = self.id[criterion]

    def show(self, fig = None, axs = None, skip = 1, **kw):
        if axs is None:
            fig, axs = plt.subplots(1, 1)
        axs.scatter(self.r[0][::skip], self.r[1][::skip], **kw)
