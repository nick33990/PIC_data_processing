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

from PIC_data_processing.xy_maps import *
from PIC_data_processing.constants import *
from PIC_data_processing.plot_utils import *


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
	return time_steps

def parse_max(val):
    if val is None:
        return [None] * 3
    elif '_' in val:
        return [float(x) for x in val.split('_')]
    else:
        return [float(val)] * 3

if __name__ == '__main__':
# palletes definition
    dataset_name = 'fields'
    viridis = add_transparency(plt.cm.viridis, lambda x:np.tanh(10*(2*x-1))  ** 2)
    s1 = add_transparency(plt.cm.seismic, lambda x:np.tanh(10*(2*x-1))  ** 2)

    args = parse_args()

    rescale = [1/args.wavelenght, '{}/lambda_L$']

    k = 2 * np.pi / args.wavelenght
    nc = wl2nc / args.wavelenght ** 2

    params_read = {
        'dataset_name' : dataset_name, 'filename' : args.filename
    }

    F = xy_map.from_openPMD(args.p, args.t[0], field = args.fields[0], from_probes = args.from_probes, **params_read)


    if args.r is None:
        ne0 = xy_map.from_openPMD(args.p, 0,\
            field = 'ne', from_probes = False, **params_read).data / nc
    else:
        ne0 = np.load(args.r) / nc

    hf_filter = 1 - SG_filter2D(F.shape, F.dx/args.wavelenght, F.dy/args.wavelenght, Rmax = 10)
    output_path = join(args.d, 'fields_lim')

    os.makedirs(output_path, exist_ok = True)

    for field in args.fields:
        for time_step in args.t:
            F = xy_map.from_openPMD(args.p, time_step, field = field, from_probes = args.from_probes, **params_read)
            ne = xy_map.from_openPMD(args.p, time_step, field = 'ne', from_probes = False, **params_read)
            if ne0.shape != ne.shape:
                print(f'!!!! ne at time step {time_step} shape ({ne.shape}) != ne0 shape ({ne0.shape})')
                sx, sy = ne0.shape[1] // ne.shape[1], ne0.shape[0] // ne.shape[0]
                ne0 = ne0[::sx, ::sy]
            ne.data = ne.data / nc - ne0
            F.center_grid()
            ne.center_grid()
            F.rescale_axis(0, 1/args.wavelenght, '$x/\lambda_L$')
            F.rescale_axis(1, 1/args.wavelenght, '$y/\lambda_L$')
            ne.rescale_axis(0, 1/args.wavelenght, '$x/\lambda_L$')
            ne.rescale_axis(1, 1/args.wavelenght, '$y/\lambda_L$')

            for i, mode in enumerate(args.modes):
                Fmin, Fmax = (None, None) if args.Emax[i] is None else (-args.Emax[i], args.Emax[i])
                nmin, nmax = (None, None) if args.nmax[i] is None else (-args.nmax[i], args.nmax[i])
                
                if mode != '1' or (mode == '2' and not '1' in modes):
                    F.apply_filter(hf_filter, inplace = True)
                if 'B' in field and not Fmin is None:
                    Fmin, Fmax = Fmin / c, Fmax / c

                fig, axs = plt.subplots(1, 3, figsize = (9, 5),\
                 gridspec_kw = {'width_ratios':[1, .05, .05]}, dpi = 200)
                F.show_overlaped(ne, fig, axs, kw1 = {'cmap':s1, 'vmin':Fmin, 'vmax':Fmax},\
                 kw2 = {'cmap':viridis, 'vmin' : nmin, 'vmax' : nmax})
                axs[0].set_aspect('equal')
                

                if mode == '2':
                    axs[0].set_xlim(args.xlim)
                    axs[0].set_ylim(args.ylim)
                
                plt.savefig(join(output_path, f'{field}_{mode}_{time_step}.png'), bbox_inches='tight')
                plt.cla()
                plt.close(fig)
