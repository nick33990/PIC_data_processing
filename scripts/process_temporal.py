import os
import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import tqdm as tqdm
from enum import Enum

from PIC_data_processing import xt_maps 
import PIC_data_processing.plot_utils as  pu 
from PIC_data_processing.constants import *
from PIC_data_processing import file_utils


class SOURCE(Enum):
    probes_circ = 'p_circ'
    probes_line = 'p_line'
    slices_dir = 's_dir'
    slices_legacy = 's_legacy'
    
    def __str__(self):
        return self.value

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p')
    parser.add_argument('-d', default = '')
    parser.add_argument('-f', '--field', default = 'Bz')
    parser.add_argument('--to_plot', default = 'hf_Ihf_xw_Iw')
    parser.add_argument('--source', default = 'p_circ', type = SOURCE, choices = list(SOURCE))
    parser.add_argument('--wavelenght', default = '3.9')
    parser.add_argument('--tlim', default = None)
    parser.add_argument('--flim', default = None)

    args = parser.parse_args()
    args.to_plot = args.to_plot.split('_')
    args.wavelenght = eval(args.wavelenght)
    args.tlim = [None, None ] if args.tlim is None else [float(x) for x in args.tlim.split('_')]
    args.flim = [None, None ] if args.flim is None else [float(x) for x in args.flim.split('_')]


    return args
    

def find_center(path):
    v = file_utils.find_params(path, 'grid', ['x_c', 'y_c', 'Ny', 'CELL_WIDTH_SI', 'CELL_HEIGHT_SI'])
    return [
        int(v['x_c'] / v['CELL_WIDTH_SI']), int(v['Ny'] - v['y_c'] / v['CELL_HEIGHT_SI'])
    ]

def circle(xy, xy_c, angle_c, R = 1250, da = 90):
    xy -= np.array(xy_c)
    x, y = xy[:, 0], xy[:, 1]
    r_sqr, theta = x * x + y * y, np.arctan2(y, x) * 180 / np.pi
    leftover = np.where((r_sqr > R * R) & (np.abs(theta - angle_c) < da))[0]


    return leftover, (theta, '$\phi^{\circ}$')

if __name__ == '__main__':
    cm = 'jet'
    args = parse_args()
    

    if args.source == SOURCE.slices_legacy:
        xt = xt_maps.xt_map.from_slices_legacy(args.p, f'{args.field}')
        xt.rescale_axis(1, (args.wavelenght) ** -1, '$x/\lambda_L$')
    elif args.source == SOURCE.probes_circ:
        param_path = ''.join([x + '\\' for x in args.p.split('\\')[:-1]])
        print(param_path)
        xy0 = find_center(param_path)
        aoi = file_utils.find_params(param_path, 'density', ['AOI_deg'])['AOI_deg']
        xt = xt_maps.xt_map.from_probes_h5(args.p, args.field, filter_fn = circle,\
            default_kw = {'xy_c':xy0, 'angle_c':90 - aoi})

    print(xt.dx, xt.dy)
    xt.rescale_axis(0, 1e-15 * (args.wavelenght * 1e-6 / 3e8) ** -1, '$t/T_L$')

    
    t0 = (np.argmax(np.sum(xt.data, axis = 0))) * xt.dx
    xt.x_origin = -t0


    fig, axs = plt.subplots(1 + len(args.to_plot), 1, figsize = (10, 6 * (1 + len(args.to_plot))))

    xt.show(axs = axs[0], cmap = cm, aspect = 'auto')
    axs[0].set_xlim(args.tlim)
    Iw, Ihf = None, None


    if True in ['xw' in a for a in args.to_plot]:
        xw = xt.xw().abs_sqr()
        xw.x_title = '$\omega/\omega_L$'
        xw.data /= np.max(xw.data)
        Iw = np.mean(xw.data, axis = 0)
    if True in ['hf' in a for a in args.to_plot]:
        hf_filter = xt_maps.SG_hf_filter1D(xt.dx, xt.data.shape[1], cent = 21, width = 8, order = 6)
        xt_hf = xt.apply_filter(hf_filter, axis = 1).abs_sqr()
        Ihf = np.mean(xt_hf.data, axis = 0)


    for i in range(1, len(axs)):
        if args.to_plot[i - 1] == 'xw':
            xw.show(axs = axs[i], log_scale = True, cmap = cm, aspect = 'auto', vmin = -5)
            axs[i].set_xlim(args.flim)
        elif args.to_plot[i - 1] == 'hf':
            xt_hf.show(axs = axs[i], cmap = cm, aspect = 'auto')
            axs[i].set_xlim(args.tlim)
        elif args.to_plot[i - 1] == 'Iw':
            axs[i].plot(xw.x_origin + np.arange(0, len(Iw)) * xw.dx, Iw)
            axs[i].set_yscale('log')
            axs[i].set_xlim(args.flim)
            axs[i].set_ylabel('$<I(\omega)>$')
        elif args.to_plot[i - 1] == 'Ihf':
            axs[i].plot(xt_hf.x_origin + np.arange(0, len(Ihf)) * xt_hf.dx, Ihf)
            axs[i].set_xlim(args.tlim)
            axs[i].set_ylabel('$<I>, a.u.$')

            
    
    plt.savefig(os.path.join(args.d, args.field + '_temporal.png'))
    plt.show()



    
