# PIC data processing
## General information
Library contains some functions, that might be useful to process and visualize of data, obtained from calculations using particle in cell method. Also some functions to load data, obtained from [PIConGPU](https://github.com/ComputationalRadiationPhysics/picongpu/tree/0.6.0) framework have been implemented.
## Brief modules description
#### file_utils.py
Contains functions that read data from output files of PIConGPU framework. From h5 file in openPMD format, and from slice printer plugin
#### time_series.py
Contains functions to plot spectra, high-frequency parts of signals and plotting isolated pulses 
#### xt_maps.py
Functions to plot arrays like $E(x,t)$ or $E(x,\omega)$ etc and perform some transforms.
#### xy_maps.py
Functions to plot 2D arrays on regular grid. And to plot two 2D arrays on same using transparency (see example)

#### constants.py
Definition of useful constants.
#### math_utils.py
Contains function to process data: performing fft, fft-filtering of signals, retrieving its envelope and tranforming 2D array from cartesian units to polar
#### plot_utils.py
Contains some auxilary functions to plot data
