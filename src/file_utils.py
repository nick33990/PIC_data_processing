import os
import h5py
import numpy as np


def get_grid_steps(m, multiplier = 1):
	ul = m['unit_length']
	dx, dy, dz = [m[attr] * ul * multiplier for attr in ['cell_width', 'cell_height', 'cell_depth']]
	return dx, dy, dz


def read_positions(f, species, two_dim = True):		
	x = np.array(f['particles'][species]['position']['x'])
	Dx = np.array(f['particles'][species]['positionOffset']['x'])

	y = np.array(f['particles'][species]['position']['y'])
	Dy = np.array(f['particles'][species]['positionOffset']['y'])

	if not two_dim:
		z = np.array(['particles'][species]['position']['z'])
		Dz=np.array(f['data'][str(time_step)]['particles'][species]['positionOffset']['z'])
	
		return (x + Dx), (y + Dy), (z + Dz)
	else:
		return (x + Dx), (y + Dy)


def density_from_fields(f, species = 'e', uc = -1.6e-19):
	dens = np.array(f['fields'][species + '_all_chargeDensity'])
	unit_si_dens = h5py.AttributeManager(f['fields'][species + '_all_chargeDensity'])['unitSI']
	return dens * unit_si_dens / uc 


# 2D



def read_probe_field(f, field, axis, ii, jj, kk = None, species = 'p'):
	F_flat = np.array(['particles'][species]['probe' + field][axis])
	F_unit_si = h5py.AttributeManager(f['particles'][species]['probe' + field][axis])['unitSI']

	two_dim = (kk is None)

	nx = len(np.unique(ii))
	ny = len(np.unique(jj))
	if two_dim:
		shape = (nx, ny)
	else:
		nz = len(np.unique(kk))
		shape = (nx, ny, nz)

	F = np.zeros(shape, dtype = 'float32')
	if two_dim:
		for i in range(len(F_flat)):
			F[ii[i], jj[i]] = F_flat[i]
	else:
		for i in range(len(F_flat)):
			F[ii[i], jj[i], kk[i]] = F_flat[i]
	return F * F_unit_si


def read_field(f, field, axis):
	F_unit_si = h5py.AttributeManager(f['fields'][field][axis])['unitSI']
	F = f['fields'][field][axis]
	return	 F_unit_si * F


def proj(s, n):
    return float(s[1:-1].split(',')[n])

projv = np.vectorize(proj)

def load_line(filename, axes, dtype = 'float32'):
    with open(filename, 'r') as f:
        line = next(iter(f))
        line = line.split(' ')[:-1]
        line = [projv(np.array(line) , axis).astype(dtype) for axis in axes]
        return line

def compress_yt(F, tr_time = 1e-5, tr_space = 1e-5):
    mean_B = np.mean(np.abs(F), axis = 1)
    start = np.where(mean_B > tr_time * np.max(mean_B))[0][0]
    F = F[start:]
    mean_B = np.mean(np.abs(F), axis = 0)
    nz = np.where(mean_B > tr_space * np.max(mean_B))[0]
    return F[:, nz[0]: nz[-1]].astype('float32')

def load_yt(path, axes, step = 2, dtype = 'float32'):
    if type(axes) == int:
        axes = [axes]
    files = sorted(os.listdir(path), key = lambda s : int(s[6:-4]))
    files = files[::step]
    lines0 = load_line(join(path, files[0]), axes, dtype = dtype)
    Ny, Nt = len(lines0[0]), len(files)
    F = [np.zeros((Nt, Ny), dtype = dtype) for _ in axes]
    for i in range(len(axes)):
        F[i][0, :] = lines0[i]
    for j, file in tqdm(enumerate(files[1:])):
        lines0 = load_line(join(path, file), axes = axes, dtype = dtype)
        for i in range(len(axes)):
            F[i][j, :] = lines0[i]
    return F