import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import RegularGridInterpolator


def FWHM(Y, level = 1 / 2, return_delta = True):
	"""
	Computes function diameted by specified level
	Parameters
	----------:
	Y - function values
	level - level to measute width at
	return delta - if True return width, else returns indices where function reaches level
	
	Returns
	----------
	width or positions where function reaches level
	"""
	half_max = np.max(Y) * level
	d = np.where(np.sign(Y - half_max) > 0)[0]
	if len(d) <= 2:
		return -1
	if return_delta:
		return d[-1] - d[0]
	else:
		return d[-1], d[0]


def Fi(I, axis = 0):
	ii = np.linspace(0, I.shape[0] - 1, I.shape[0])
	phase_factor = np.exp(-1j * np.pi * ii)
	if len(I.shape) == 2:
		phase_factor = phase_factor.reshape((-1, 1) if axis == 0 else (1, -1))
	ift = np.fft.ifft(phase_factor * I, axis = axis)
	return np.array( phase_factor * ift).reshape(I.shape)

def F(I, axis = 0):
	ii = np.linspace(0, I.shape[axis] - 1, I.shape[axis])
	phase_factor = np.exp(-1j * np.pi * ii)
	if len(I.shape) == 2:
		phase_factor = phase_factor.reshape((-1, 1) if axis == 0 else (1, -1))

	ift = np.fft.fft(I * phase_factor, axis = axis)
	return np.array( ift * phase_factor).reshape(I.shape)

def F2(I):
	ii = np.linspace(0, I.shape[0] - 1, I.shape[0])
	jj = np.linspace(0, I.shape[1] - 1, I.shape[1])
	x,y = np.meshgrid(ii, jj, sparse=True)
	phase_factor = np.exp(-1j * np.pi * (x + y)).T

	ift = np.fft.fft2(phase_factor * I)
	return np.array( phase_factor * ift).reshape(I.shape)

def Fi2(I):
	ii = np.linspace(0, I.shape[0] - 1, I.shape[0])
	jj = np.linspace(0, I.shape[1] - 1, I.shape[1])
	x,y = np.meshgrid(ii, jj, sparse=True)
	phase_factor = np.exp(-1j * np.pi * (x + y)).T

	ift = np.fft.ifft2(phase_factor * I)
	return np.array( phase_factor * ift).reshape(I.shape)

def fft_filter_F(f, pass_filter, axis = 0):
	"""
	Performs spectral filtering of signal

	Parameters
	----------:
	f - signal to be filtered
	pass_filter - transmission of spectral filter 
	
	Returns
	----------
	filtered signal
	"""
	fw = F(f, axis = axis)
	fw = fw * pass_filter
	return np.real(Fi(fw, axis = axis))

def fft2_filter_F(A, mask):
	fw = F2(A)
	fw = fw * mask
	return np.real(Fi2(fw))


def env_pos_freq(sl):
	"""
	retrieves signal envelope by filtering only positive frequencies
	
	Parameters
	----------:
	s - signal to retrieve envelope
	Returns:
	env - signal envelope

	Returns
	----------
	envelope of signal
	"""
	fw = F(sl)
	n = len(sl)
	fw[:len(fw)//2] *= 0
	env = 4 * np.abs(Fi(fw)) ** 2

	return env


def chirp(t, A = 1,t0 = 0.0, T = 0.1, f = 30, f2 = 120, CEP = 0):
   	return A * (np.exp(-0.5 * ((t - t0) / T) ** 2) * np.cos(f * t + f2 * t ** 2 + CEP) )

def env_fit(s, waveform = chirp, p0 = [2, -0.1, 0.2, 20, 20, 0],\
		   adj = [1, 1]):

	"""
	retrieves signal enveolope by fitting it to certain waveform
	Parameters
	----------:
	s - signal to retrieve envelope
	waveform - function to fit signal. first argument - time,
	 folowing - are params to be fit - amplitude, time offset, pulse duration etc.
	p0 - initial guess of parametes to fin
	adj - adjustemt of params 

	Returns
	----------
	envelope of signal
	"""

	s /= np.max(s)
	t_au = np.linspace(-1, 1, len(s))

	p = curve_fit(waveform, t_au, s, p0 = p0)
	p = p[0]
	p_rec = p.copy()
	p[3:] *= 0
	p[0] = 1
	appr = adj[0] * waveform(t_au / adj[1], *p) ** 2 
	return appr, p_rec

def env_conv(s, kernel):
	"""
	retrieves signal enveolope by convolution with kernel

	Parameters
	----------:
	s - signal to retrieve envelope
	kernel - to convolve s with

	Returns
	----------
	envelope of signal
	"""
	I = s ** 2
	Imax = np.max(I)
	env = np.convolve(I / Imax, kernel, mode = 'same')
	env /= np.sum(kernel)
	return env


def cart2pol_interp(x, y, A):
	"""
	transforms array on cartesian grid A(x, y) to polar grid A(r, theta).

	Parameters
	----------
	x, y - arrays of coordinates
	A - array to be transformed
	
	Returns
	----------
	r, theta - arrays or polar coordinates r = sqrt(x^2 + y^2) and angles (in degrees),
	A_ra - array in polar coordinates
	"""
	Nx, Ny = A.shape

	borders_x = np.min(x), np.max(x)
	borders_y = np.min(y), np.max(y)
	interp = RegularGridInterpolator((x, y), A, bounds_error = False, fill_value = 0)
	kr, theta = np.linspace(0, np.sqrt(x[-1] ** 2 + y[-1] ** 2), len(x)),\
				np.linspace(0, 2 * np.pi, len(y)) # len(x)
	kr, theta = np.meshgrid(kr, theta)
	Kx, Ky = kr * np.sin(theta), kr * np.cos(theta)
	coords = np.vstack((Kx.flatten(), Ky.flatten())).T
	theta *= 180 / np.pi
	return kr[0], theta[:, 0], interp(coords).reshape(Kx.shape)[::-1]