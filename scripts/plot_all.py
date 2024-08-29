import os
import sys
import h5py
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import windows
import argparse
from tqdm import tqdm
from time import time


from PIC_data_processing.math_utils import F2, Fi2, fft2_filter_F
from PIC_data_processing.file_utils import read_field, density_from_fields, get_grid_steps
from PIC_data_processing.xy_maps import plot_xy

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

class Mode:
	RAW = 0
	FFT_filtered = 1
	FFT_filtered_zoomed = 2

if __name__ == '__main__':
	path = '../CWE_old/output-run/simOutput/openPMD'
	parser = argparse.ArgumentParser(
					prog='plot_fields.py',
					description='plot_specified_fields')
	parser.add_argument('-t', default = sorted([int(f[8:-3]) for f in os.listdir(path)]))
	parser.add_argument('-d', default = '')

	args = parser.parse_args()
	time_steps = args2time_steps(args.t)

# 	print(time_steps)
# # params
# 	exit()
	wavelenght = 3.9
	dpi = 500
	fields = [['E', 'x'], ['E', 'y'], ['B', 'z']]
	k = 2 * np.pi / wavelenght

	t0 = time()
	nc = 1.142e27 / (wavelenght ** 2)

	vmin, vmax = -5, 5
	il, jl = 3200, 3200
	modes = [Mode.RAW, Mode.FFT_filtered, Mode.FFT_filtered_zoomed]
	figsize = (11, 8)
	field_cmap = 'jet'
	dens_cmap = 'viridis'
	interpolation = 'bicubic'
	najor = 1

	# if not os.path.exists(dest_path) and len(dest_path) > 0:
	# 	os.mkdir(dest_path)
	dest_path = os.path.join(args.d, 'fields_full_scale')
	if not os.path.exists(dest_path):
		os.mkdir(dest_path)

# dens ref	
	if os.path.exists(os.path.join(path, 'simData_{:06d}.h5'.format(0))):
		f = h5py.File(os.path.join(path, 'simData_{:06d}.h5'.format(0)), 'r')
		f = f['data'][str(0)]
		dens_ref = density_from_fields(f, 'e')[:il, :jl] / nc 
		# np.save('dens_ref')
		print(np.max(dens_ref))
	else:
		dens_ref = 0



	for time_step in tqdm(time_steps):

		f = h5py.File(os.path.join(path, 'simData_{:06d}.h5'.format(time_step)), 'r')
		f = f['data'][str(time_step)]
		m = h5py.AttributeManager(f)

		for nn in modes:
			fig, axs = plt.subplots(2, 4, figsize = figsize, dpi = 200, \
					   gridspec_kw = {'width_ratios' : [1, 0.05, 1, 0.05],\
									 'height_ratios' : [1, 1],
									 'wspace':0.05
									 })

			for i, field in enumerate(fields + ['ne']):
				Field= read_field(f, field[0], field[1])[:il, :jl] if i < len(fields) else\
				 density_from_fields(f, 'e')[:il, :jl] / nc - (0 if nn == 0 else dens_ref)
############################ fft filter init #################################
				if time_step == time_steps[0] and i == 0 and nn == modes[0]:
					dx, dy, dz = get_grid_steps(m, 1e6)
					Ny, Nx = Field.shape
					X, Y =  np.linspace(0, Nx * dx, Nx),\
							np.linspace(0, Ny * dy, Ny)
					X, Y = (X - X[Nx // 2]) / wavelenght, (Y - Y[Ny // 2]) / wavelenght
					Kx, Ky= np.linspace(-1, 1, Nx) * np.pi / dx,\
							np.linspace(-1, 1, Ny) * np.pi / dy

					dk = Ky[1] - Ky[0]
					mask_power = 6
					mask_width = 12 * dk
					Kx, Ky = np.meshgrid(Kx, Ky)

					w = windows.hann(Ny)[None, :].T * windows.hann(Nx)[None, :]#make_window(Nx)
					mask = (1 - np.exp(-(((Kx * Kx + Ky * Ky) / (k + mask_width) ** 2)) ** mask_power))

					########
				if nn != 2:
					xl = (X[0], X[-1])
					yl = (Y[0], Y[-1])
				else:
					xl = (-1.5, 1.5)
					yl = (-1.5, 1.5)

#############################################################################
				if nn > 0 and i < 3:
					Field= fft2_filter_F(Field* w, mask)


############################### plot ########################################
				im = [[None, None], [None, None]]
				if i < 3:
					lims = najor * min(-np.min(Field), np.max(Field)) 
					lims = (-lims, lims)
				else:
					lims = (vmin, vmax)

				ii, jj = i // 2, 2 * (i % 2)

				plot_xy(fig, axs[ii, jj:jj + 2], Field, X, Y, xlim = xl, ylim = yl,\
					imshow_kwargs = {'cmap':field_cmap if i < 3 else dens_cmap,\
					 'vmin':lims[0], 'vmax':lims[1]})
				axs[ii, jj].set_aspect('equal')
				axs[ii, jj].invert_yaxis()
				if jj == 0:
					axs[ii, jj].set_ylabel('$y/\lambda_L$')
				else:
					axs[ii, jj].set_yticks([])
				if ii == 1:
					axs[ii, jj].set_xlabel('$x/\lambda_L$')
				else:
					axs[ii, jj].set_xticks([])

			axs[0][0].set_title('$E_x$')
			axs[0][2].set_title('$E_y$')
			axs[1][0].set_title('$B_z$')
			axs[1][2].set_title('$n_e - n_{e0}$'\
			 if not isinstance(dens_ref, int) else '$n_e$')
			fig.suptitle('t = {} fs'.format(int(m['unit_time'] * 1e15 * time_step)))
			plt.savefig(os.path.join(dest_path, '{}_{}.png'.format(nn, time_step)))
			plt.cla()
			plt.close(fig)
