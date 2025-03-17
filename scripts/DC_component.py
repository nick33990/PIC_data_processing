import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from os.path import join
from tqdm import tqdm
import imageio
import argparse

from PIC_data_processing.plot_utils import *
from PIC_data_processing.xy_maps import *
from PIC_data_processing.constants import *

def args2timesteps(time_steps):
    if type(time_steps) is list:
        return time_steps
    time_steps = time_steps.split('..')
    if len(time_steps) > 1:
        t_start, t_end, t_per = int(time_steps[0]), int(time_steps[1]), int(time_steps[2])
        time_steps = range(t_start, t_end, t_per)
    else:
        time_steps = [int(time_steps[0])]

    return time_steps


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t')
    parser.add_argument('-p')
    parser.add_argument('-d', default = '')
    parser.add_argument('-w', '--wavelenght', default = '3.9')
    parser.add_argument('--from_probes', action = 'store_true')
    parser.add_argument('--pad_array', action = 'store_true')
    parser.add_argument('--field', default = 'Bz')
    parser.add_argument('--Emax', default = None)
    parser.add_argument('--nmax', default = None)
    parser.add_argument('--xlim', default = '-2_2')
    parser.add_argument('--ylim', default = '-2_2')
    parser.add_argument('--species', default = 'p')
    parser.add_argument('--tcenter', default = '26000')

    args = parser.parse_args()

    args.t = args2timesteps(args.t)
    
    args.wavelenght = float(args.wavelenght)
    args.Emax = float(args.Emax) if not args.Emax is None else None
    args.nmax = float(args.nmax) if not args.nmax is None else None
    args.tcenter = float(args.tcenter)
    args.xlim = [float(x) for x in args.xlim.split('_')]
    args.ylim = [float(x) for x in args.ylim.split('_')]

    return args

# def find_center(args.p):
#     v = file_utils.find_params(args.p, 'grid', ['x_c', 'y_c', 'Ny', 'CELL_HEIGHT_SI'])
#     return [
#         (float(v['x_c']) ), ( float(v['y_c']))
#     ]

if __name__ == '__main__':
    args = parse_args()

    viridis = add_transparency(two_color(np.array([1,1,1]), np.array([0,0,0])), lambda x:0.0)
    s1 = add_transparency(plt.cm.seismic, lambda x:np.tanh(10*(2*x-1))  ** 2)
    
    filename = os.listdir(args.p)[0]
    ext = filename.split('.')[-1]
    filename = '_'.join(filename.split('_')[:-1]) + '_'
    files = [join(args.p, f'{filename}{t:06d}.{ext}') for t in args.t]

    args.d = join(args.d, f'DC_{args.field}')

    TL = args.wavelenght * um / c
    nc = wl2nc / (args.wavelenght ** 2)

    if not os.path.exists(args.d):
        os.mkdir(args.d)

    with h5py.File(files[0], 'r') as f:
    
        F = xy_map.from_openPMD(files[0], field = args.field,\
         from_probes = args.from_probes,  species = args.species)
        F.rescale_axis([0, 1], 1 / args.wavelenght, ['$x/\lambda$', '$y/\lambda$'])

        ne0 = xy_map.from_openPMD(files[0],\
            field = 'ne', from_probes = False)
        ne0.rescale_axis([0, 1], 1 / args.wavelenght, ['$x/\lambda$', '$y/\lambda$'])
        ne0.crop_b(*args.xlim, *args.ylim, inplace = True)

        F.crop_b(*args.xlim, *args.ylim, inplace = True)

        wy, wx = F.data.shape
        
        m = h5py.AttributeManager(f['data'][str(args.t[0])])
        dt_sim = m['unit_time'] / TL
        dt_save = (args.t[1] - args.t[0])
        dt = dt_sim * dt_save
        Nbuff = int(4 / dt)
        Nbuff = (Nbuff // 2 + 1) * 2
        t = np.arange(0, Nbuff) * dt  #[e^-4]
        w = np.exp(-((t - t[len(t) // 2]) / 1) ** 2)
        w /= np.sum(w)
        buff = np.zeros((Nbuff, wy, wx))



    for i in tqdm(range(len(args.t) + Nbuff * int(args.pad_array)), total = len(args.t) + Nbuff* int(args.pad_array)):
            
        if i < len(args.t):
            F = xy_map.from_openPMD(files[i], field = args.field,\
                    from_probes = args.from_probes, species = args.species)

            F.rescale_axis([0, 1], 1 / args.wavelenght, ['$x/\lambda$', '$y/\lambda$'])
            F.crop_b(*args.xlim, *args.ylim, inplace = True)
            buff[i % Nbuff] = F.data
        else:
            buff[i % Nbuff] = np.zeros((wy, wx))
        DC = np.sum(buff * w.reshape(-1, 1, 1), axis = 0) * MGs
        w = np.roll(w, 1)


        if i >= Nbuff // 2 and i < len(args.t) + Nbuff //2:
            ne = xy_map.from_openPMD(files[i - Nbuff // 2], field = 'ne',)
            ne.rescale_axis([0, 1], 1 / args.wavelenght, ['$x/\lambda$', '$y/\lambda$'])
            ne.crop_b(*args.xlim, *args.ylim, inplace = True)

            ne = (ne - ne0) / nc
            F.data = DC
            fig, axs = plt.subplots(1, 1)#, gridspec_kw = {'width_ratios':[1, .05, .05], 'wspace':0.5})
            axs = [axs]

            if (i - Nbuff // 2) % 5:
                np.save(f'{args.d}/{args.t[i - Nbuff // 2]}', F.data)
            F.show_overlaped(ne, fig = fig, axs = axs, kw1 = {'cmap':s1, 'vmin':-2, 'vmax':2},\
                            kw2 = {'cmap':viridis, 'vmin' : None, 'vmax' : None})
            plt.title('$' + f'{args.field[0]}_{args.field[1]}' + '$' +\
                 ', $t-t_{max}$=' + f'{round((args.t[i - Nbuff // 2] - args.tcenter) * TL * dt_sim * 1e15)} fs')
            plt.savefig(f'{args.d}/{args.t[i - Nbuff // 2]}.png', bbox_inches = 'tight')  
            plt.close(plt.gcf())
            plt.cla()
