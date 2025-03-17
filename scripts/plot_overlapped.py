import os
import cv2
import argparse
import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os.path import join
from collections.abc import Iterable
from pathlib import Path
from tqdm import tqdm
from time import time
from enum import Enum

from PIC_data_processing.xy_maps import *
from PIC_data_processing.constants import *
from PIC_data_processing.plot_utils import *

class FILTER_TYPE(Enum):
    HHG = 'HHG'
    THz = 'THz'

    def __str__(self):
        return self.value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t')
    parser.add_argument('-p')
    parser.add_argument('-d', default = '')
    parser.add_argument('-r', default = None)
    parser.add_argument('-w', '--wavelenght', default = '3.9')
    parser.add_argument('--filename', default = 'simData_')
    parser.add_argument('--from_probes', action = 'store_true')
    parser.add_argument('--fields', default = 'Ey_Bz')
    parser.add_argument('--modes', default = '012')
    parser.add_argument('--Emax', default = None)
    parser.add_argument('--nmax', default = None)
    parser.add_argument('--xlim', default = '-2_2')
    parser.add_argument('--ylim', default = '-2_2')
    parser.add_argument('--species', default = 'p')
    parser.add_argument('--filter_type', default = 'HHG', type = FILTER_TYPE, choices = list(FILTER_TYPE))

    args = parser.parse_args()

    args.t = args2time_steps(args.t)

    args.wavelenght = float(args.wavelenght)
    args.fields = args.fields.split('_')
    args.Emax = parse_max(args.Emax)
    args.nmax = parse_max(args.nmax)
    args.xlim = [float(x) for x in args.xlim.split('_')]
    args.ylim = [float(x) for x in args.ylim.split('_')]

    return args


def args2time_steps(time_steps):
	if type(time_steps) is list:
		return time_steps
	time_steps = time_steps.split('..')
	if len(time_steps) > 1:
		t_start, t_end, t_per = int(time_steps[0]), int(time_steps[1]), int(time_steps[2])
		time_steps = range(t_start, t_end, t_per)
	else:
		time_steps = [int(time_steps[0])]
	print(time_steps)
	return time_steps

def parse_max(val):
    if val is None:
        return [None] * 3
    elif '_' in val:
        return [float(x) for x in val.split('_')]
    else:
        return [float(val)] * 3

if __name__ == '__main__':
    dataset_name = 'fields'
    viridis = add_transparency(plt.cm.viridis, lambda x:np.tanh(10*(2*x-1))  ** 2)
    s1 = add_transparency(plt.cm.seismic, lambda x:np.tanh(10*(2*x-1))  ** 2)

    args = parse_args()

    filename = os.listdir(args.p)[0]
    ext = filename.split('.')[-1]
    filename = '_'.join(filename.split('_')[:-1]) + '_'
    files = [join(args.p, f'{filename}{t:06d}.{ext}') for t in args.t]


    k = 2 * np.pi / args.wavelenght
    nc = wl2nc / args.wavelenght ** 2

    params_read = {
        'dataset_name' : dataset_name, 'species' : args.species
    }

    F = xy_map.from_openPMD(files[0], field = args.fields[0], from_probes = args.from_probes, **params_read)


    if args.r is None:
        if args.filter_type == FILTER_TYPE.HHG:
            ne0 = xy_map.from_openPMD(files[0],\
                field = 'ne', from_probes = False, **params_read).data / nc
        else:
            ne0 = 0
    else:
        ne0 = np.load(args.r) / nc

    kx, ky = np.linspace(-0.5, 0.5, F.shape[1]) * 2 * pi / F.dx / k,\
            np.linspace(-0.5, 0.5, F.shape[0]) * 2 * pi / F.dy / k
    Kx, Ky = np.meshgrid(kx, ky)
    Dw = 1
    kmax = 21 / 2
    if args.filter_type == FILTER_TYPE.HHG:
        Kr = np.sqrt(Kx ** 2 + Ky ** 2)
        hf_filter = 1 - np.exp(-(0.5 * (Kr - kmax) / (.5 * (7  + 1.0 * Dw))) ** 6)
        hf_filter[np.where(Kr > kmax)] = 0
        hf_filter = 1 - hf_filter
    elif args.filter_type == FILTER_TYPE.THz:
        filter_width = 0.5
        filter_pow = 6
        hf_filter = np.exp(-((Kx ** 2 + Ky ** 2) / filter_width ** 2) ** filter_pow)
    else:
        hf_filter = 1


    output_path = join(args.d, 'fields')

    os.makedirs(output_path, exist_ok = True)

    for field in args.fields:
        for i, time_step in tqdm(enumerate(args.t)):
            F = xy_map.from_openPMD(files[i], field = field, from_probes = args.from_probes, **params_read)
            ne = xy_map.from_openPMD(files[i], field = 'ne', from_probes = False, **params_read)


            if hasattr(ne0, 'shape') and ne0.shape != ne.shape:
                print(f'!!!! ne at time step {time_step} shape ({ne.shape}) != ne0 shape ({ne0.shape})')
                sx, sy = ne0.shape[1] // ne.shape[1], ne0.shape[0] // ne.shape[0]
                ne0 = ne0[::sx, ::sy]
            ne = ne / nc - ne0
            F.center_grid()
            ne.center_grid()
            F.rescale_axis([0, 1], 1/args.wavelenght, ['$x/\lambda_L$', '$y/\lambda_L$'])
            ne.rescale_axis([0, 1], 1/args.wavelenght, ['$x/\lambda_L$', '$y/\lambda_L$'])

            for i, mode in enumerate(args.modes):
                Fmin, Fmax = (None, None) if args.Emax[i] is None else (-args.Emax[i], args.Emax[i])
                nmin, nmax = (None, None) if args.nmax[i] is None else (-args.nmax[i], args.nmax[i])

                if mode == '1' or (mode == '2' and not '1' in args.modes):
                    F.apply_filter(hf_filter, inplace = True)
                    absorber = 64
                    F.data[:absorber] = 0
                    F.data[-absorber:] = 0
                    F.data[:, :absorber] = 0
                    F.data[:, -absorber:] = 0


                if 'B' in field and not Fmin is None:
                    Fmin, Fmax = Fmin / c, Fmax / c

                fig, axs = plt.subplots(1, 1, figsize = (10, 5), dpi = 400)

                axs = [axs]

 
                F.show_overlaped(ne, fig, axs, kw1 = {'cmap':s1, 'vmin':Fmin, 'vmax':Fmax},\
                 kw2 = {'cmap':viridis, 'vmin' : nmin, 'vmax' : nmax})

                axs[0].set_aspect('equal')

                if mode == '2':
                    axs[0].set_xlim(args.xlim)
                    axs[0].set_ylim(args.ylim)
                
                plt.savefig(join(output_path, f'{field}_{mode}_{time_step}.png'))#, bbox_inches='tight')
                plt.cla()
                plt.close(fig)

