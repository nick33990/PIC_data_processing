# PIC data processing
## General information
Library contains some functions, that might be useful to process and visualize of data, obtained from calculations using particle in cell method. Also some functions to load data, obtained from [PIConGPU](https://github.com/ComputationalRadiationPhysics/picongpu/tree/0.6.0) framework have been implemented.
## Brief modules description
#### file_utils.py
Contains functions that read data from output files of PIConGPU framework. From h5 file in openPMD format, and from slice printer plugin
#### math_utils.py
Contains function to process data: performing fft, fft-filtering of signals, retrieving its envelope and tranforming 2D array from cartesian units to polar
#### plot_utils.py
Contains some auxilary functions to plot data
#### time_series.py
Contains functions to plot spectra

High-frequency parts of signals

And plotting isolated pulses 
#### xt_maps.py
Functions to plot arrays like $E(x,t)$ or $E(x,\omega)$ etc
#### xy_maps.py
#### constants.py
