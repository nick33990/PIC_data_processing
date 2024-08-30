import os
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from os.path import join
from scipy.ndimage import zoom
from scipy.signal import windows
from matplotlib.colors import ListedColormap
import argparse
from tqdm import tqdm

from PIC_data_processing.file_utils import read_field, get_grid_steps, density_from_fields
from PIC_data_processing.plot_utils import two_color, with_white, add_transparency
from PIC_data_processing.xy_maps import plot_overlap_maps
from PIC_data_processing.math_utils import fft2_filter_F

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

if __name__ == '__main__':
	path = '../CWE_old/output-run/simOutput/openPMD'
	parser = argparse.ArgumentParser(
					prog='plot_fields.py',
					description='plot_specified_fields')
	parser.add_argument('-t', default = sorted([int(f[8:-3]) for f in os.listdir(path)]))
	parser.add_argument('-d', default = '')

	args = parser.parse_args()
	time_steps = args2time_steps(args.t)

	wavelenght = 3.9 # all units are um
	nc = 1.142e27 / (wavelenght ** 2)
	scale_factor = 1
	stretch_filter = True
	subtract_ref = stretch_filter
	if subtract_ref:
		with h5py.File(os.path.join(path, 'simData_{:06d}.h5'.format(0)), 'r') as f:
			f = f['data']['0']
			ne_ref = density_from_fields(f) / nc
			ne_ref = zoom(ne_ref, scale_factor)
	else:
		ne_ref = 0

	grid_defined = False

	viridis = with_white(plt.cm.viridis).reversed() 
	if not subtract_ref:
		viridis = ListedColormap(viridis(np.arange(128, 255)))
	rw = two_color(np.array([1,1,1]), np.array([1,0,0]),0.75)
	rw1 = add_transparency(rw, lambda x: 1 / (1 + np.exp((-0.1+x) * -50)))
	vmin, vmax = [-1.5, 1.5] if subtract_ref else [0, 10]
	dest_path = os.path.join(args.d, 'ne_Bz')
	if not os.path.exists(dest_path):
		os.mkdir(dest_path)

	for time_step in tqdm(time_steps):
		with h5py.File(os.path.join(path, 'simData_{:06d}.h5'.format(time_step)), 'r') as f:
			f = f['data'][str(time_step)]
			m = h5py.AttributeManager(f)
			dx, dy, dz = get_grid_steps(m, 1e6)
			dt = m['unit_time']

			B = read_field(f, 'B', 'z')
			B = zoom(B, scale_factor)

			ne = zoom(density_from_fields(f) / nc, scale_factor)

		if not grid_defined:
			grid_defined = True
			k = 2 * np.pi / wavelenght

			Ny, Nx = B.shape
			X, Y =  np.linspace(0, Nx * dx, Nx),\
			np.linspace(0, Ny * dy, Ny)
			X, Y = (X - X[Nx // 2]) / wavelenght,\
					(Y - Y[Ny // 2]) / wavelenght
			Kx, Ky = np.linspace(-1, 1, Nx) * np.pi / dx,\
								np.linspace(-1, 1, Ny) * np.pi / dy
			dk = Ky[1] - Ky[0]
			mask_power = 4
			mask_width = 12 * dk
			Kx, Ky = np.meshgrid(Kx, Ky)
			if not stretch_filter: #is_ROM:
				mask = (1 - np.exp(-(((Kx * Kx + Ky * Ky) / (k + mask_width) ** 2)) ** mask_power))

			else:
				angle = np.pi / 4  
				Kx_rot, Ky_rot = (Kx * np.cos(angle) + Ky * np.sin(angle)),\
						 (Ky * np.cos(angle) - Kx * np.sin(angle))
				mask = 1 - np.exp(-((( (Kx_rot * Kx_rot) / (k + 1 * mask_width) ** 2) +\
						  (Ky_rot * Ky_rot) / (k + 20 * mask_width) ** 2)) ** mask_power)
			w = windows.hann(Ny)[None, :].T * windows.hann(Nx)[None, :]

		B = fft2_filter_F(B * w, mask)

		fig, axs = plt.subplots(1, 3, figsize = (8, 5),\
						gridspec_kw = {'width_ratios':[1, 0.05, 0.05], 'wspace':0.7},)

		plot_overlap_maps(fig, axs, ne - ne_ref, (B/np.max(B))**2, X, Y,\
				  imshow1_kwargs = {'cmap':viridis, 'vmin':vmin, 'vmax':vmax, 'interpolation':'bicubic'},\
				   imshow2_kwargs = {'cmap':rw1}, xlim = [-1.5, 1.5], ylim = [-1.5, 1.5])
		axs[0].set_aspect('equal')
		axs[0].invert_yaxis()
		axs[0].set_xlabel(r'$x/\lambda_L$')
		axs[0].set_ylabel(r'$y/\lambda_L$')
		axs[1].set_title('$(n_e-n_c)/n_c$' if subtract_ref else '$n_e/n_c$')
		axs[2].set_title('$B_z$')
		title = '{}_fs.png'.format(int(time_step * dt * 1e15))
		fig.suptitle(title)

		plt.savefig(join(dest_path, title))