import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Iterable

from PIC_lib.math_utils import F, Fi, fft_filter_F, FWHM

def _getlims(lim, step):
	lower = int(lim[0] / step) if not lim[0] is None else 0
	upper = int(lim[1] / step) + 1 + 1 if not lim[1] is None else -1
	return slice(lower, upper, 1)

def plot_spectra(fig, axs, ft_list, dt,\
	unit_x = [r'$\omega$, Гц', 1], unit_y = ['I, отн.ед.', 'max'],\
	W = 1, yscale = 'log', titles = None,\
	xlim = [None, None], ylim = [None, None], \
 	process_axes = lambda x:x, plot_kwargs = {}):
	"""
	Plots spectra of signals at one figure
	fig - matplotlib figure object
	axs - matplotlib axes object
	ft_list - temporal dynamics 
	dt - time step of signal
	unit_x - list of title of x-axis and norm with respect to dw
	unit_y - list of title of y-axis and norm with respect to values of spectra
	W - spectral window
	yscale - scale of y-axis
	titles - names of each subplot
	xlim - limits of x-axis
	ylim - limits of y-axis
	process_axes - function, which is applied to every axes and modifies it
	plot_kwargs - addition parameters to plot spectra
	"""
	if not isinstance(dt, Iterable):
		dt = [dt] * len(ft_list)


	for i, ft in enumerate(ft_list):
		Nt = len(ft)
		w = np.linspace(0, 0.5, Nt // 2) * 2 * np.pi / dt[i] / unit_x[1]
		dw = w[1] - w[0]

		fw = F(ft * W)[Nt // 2:]
		fw_amp, fw_ph = np.abs(fw), np.angle(fw)
		if unit_y[1] == 'max':
			fw_amp /= np.max(fw_amp)
		else:
			fw_amp /= unit_y[1]

		idx_to_plot = _getlims(xlim, dw)	
		axs[i].plot(w[idx_to_plot], fw_amp[idx_to_plot], **plot_kwargs)
		axs[i].set_yscale(yscale)

		axs[i].set_xlim(xlim)	
		axs[i].set_ylim(ylim)
		process_axes(axs[i])
		if not titles is None:
			axs[i].set_title(titles[i])
	axs[len(axs) // 2].set_ylabel(unit_y[0])
	axs[-1].set_xlabel(unit_x[0])
	

def plot_HF_part(fig, axs, ft_list, dt, pass_filter_func,\
 	titles = None, unit_x = ['t, с', 1], unit_hf = ['I, усл.ед.', 1],\
 	unit_full = ['E, В/м', 1], xlim = [None, None], hf_line_kwargs = {},\
 	full_line_kwargs = {'ls' : '--', 'c' : 'k', 'lw' : 0.4}, sqr_hf = True,\
 	process_axes = lambda x:x):
	"""
	Plots full signal and its high-frequency part at one figure
	fig - matplotlib figure object
	axs - matplotlib axes object
	ft_list - temporal dynamics 
	dt - time steps of each element in ft_list
	pass_filter_func - function, which gets anguler frequency (in 1/dt units) array and returns values of pass filter
	titles - name of each element in ft_list
	unit_x - list of title of x-axis and norm with respect to dt
	unit_hf - list of title of high-frequency part and norm with respect to values of temporal dynamics
	unit_full - list of title of full signal and norm with respect to values of temporal dynamics
	xlim - minimum and maximum values of x-axis
	hf_line_kwargs - parameters of hf part plot
	full_line_kwargs - parameters of full signal plot
	sqr-hf - if raise hf part to square
	process_axes - function, which is applied to every axes and modifies it
	"""
	if not isinstance(dt, Iterable):
		dt = [dt] * len(ft_list)

	ft_hf_list = []
	for i, ft in enumerate(ft_list):
		Nt = len(ft)
		w = np.linspace(-0.5, 0.5, Nt) * 2 * np.pi / dt[i]
		pass_filter = pass_filter_func(w)

		t = np.arange(0, Nt) * dt[i] / unit_x[1]

		ft_hf = fft_filter_F(ft, pass_filter)
		if unit_hf[1] == 'max':
			ft_hf /= np.max(ft_hf)
		else:
			ft_hf /= unit_hf[1]

		ft_hf_list.append(ft_hf.copy())

		sl = _getlims(xlim, dt[i] / unit_x[1])

		axs[i].plot(t[sl], ft_hf[sl] ** (1 + int(sqr_hf)), label = 'Высокочастотный отклик', **hf_line_kwargs)
		axs[i].plot([t[sl][0]], [ft_hf[sl][0] / unit_hf[1]], label = 'Полный отклик', **full_line_kwargs)
		if i == 0:
			plt.figlegend(loc = 'upper center', ncol = 2)
			# axs[i].legend(loc = (0.2, 1.4), ncol = 2)
		if i == len(ft_list) - 1:
			axs[i].set_xlabel(unit_x[0])

		process_axes(axs)
		axs[i].set_xlim(xlim)
		# axs[i].set_ylim(ylim)
		ax = axs[i].twinx()
		ax.plot(t[sl], ft[sl] / unit_full[1], **full_line_kwargs)
		if i == len(ft_list) // 2:
			axs[i].set_ylabel(unit_hf[0])
			ax.set_ylabel(unit_full[0])
		ax.set_xlim(xlim)
		if not titles is None:
			axs[i].set_title(titles[i])


	return ft_hf_list


def plot_pulses(fig, axs, ft_list, dt, lenght_to_show, envelope_function, \
 	titles = None, unit_x = ['t, с', 1],\
 	unit_y = ['E, В/м', 1], sqr_pulse = True, pulse_kwargs = {},\
 	 envelope_kwargs = {'ls' : '--', 'c' : 'k', 'lw' : 0.4}):
	"""
	Plots most intense pulse of sequence
	fig - matplotlib figure object
	axs - matplotlib axes object
	ft_list - temporal dynamics 
	dt - time steps of each element in ft_list
	lenght to show - part of each time series to show (same units as dt)
	envelope_function - function, that extracts envelope
	titles - name of each element in ft_list
	unit_x - list of title of x-axis and norm with respect to dt
	unit_y - list of title of pulse and norm with respect to values of temporal dynamics
	sqr_pulse - plot squared pulse
	pulse_kwargs - parameters to plot pulse
	envelope_kwargs - parameters to plot envelope
	"""
	if not isinstance(dt, Iterable):
		dt = [dt] * len(ft_list)
	for i, ft in enumerate(ft_list):
		Nt = len(ft)
		t = np.arange(0, Nt) * dt[i] / unit_x[1]
		idx = np.argmax(ft)
		xlim = [t[idx] - lenght_to_show, t[idx] + lenght_to_show]

		sl = _getlims(xlim, dt[i] / unit_x[1])
		env = envelope_function(ft[sl])
		if not sqr_pulse:
			env = np.sqrt(env)
		ft[sl] = ft[sl] ** (1 + int(sqr_pulse)) / unit_y[1]
		axs[i].plot(t[sl], ft[sl], **pulse_kwargs)


		lenght = FWHM(env, 1 / 2) * dt[i] * 1e18 #* dt * 1e18
		if lenght < 3000:
			axs[i].text(t[sl][int(0.4 * (lenght_to_show))], np.max(ft) * 0.8, r'$FWHM =' + f' {round(lenght, 3)}$ aс')
			axs[i].plot(t[sl], env, **envelope_kwargs) 
		if not titles is None:
			axs[i].set_title(titles[i])


# path = '/home/labuser/rundir/old_results'
# dirs = [join(path, x) for x in os.listdir(path)][2:3]
# for dir_ in dirs:
# 	print(dir_)
# 	content = os.listdir(dir_)
# 	h5_files = [x for x in content if '.h5' in x]
# 	if len(h5_files) == 0:
# 		continue
# 	os.mkdir(dir_ + '_h5')
# 	for file in h5_files:
# 		print(join(dir_, file))