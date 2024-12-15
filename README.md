# PIC data processing
### General information
Library contains some functions, that might be useful to process and visualize of data, obtained from calculations using particle in cell method. Also some functions to load data, obtained from [PIConGPU](https://github.com/ComputationalRadiationPhysics/picongpu/tree/0.6.0) framework have been implemented. 

For examples see scripts folder.

### Installation
<!-- ```pip install -i https://test.pypi.org/simple/ PIC-data-processing``` -->
```python3 -m pip install git+https://github.com/nick33990/PIC_data_processing.git```
### Brief modules description
* **file_utils.py**:
Contains functions that read data from output files of PIConGPU framework. From h5 file in openPMD format, and from slice printer plugin
* **time_series.py**:
Contains functions to plot spectra, high-frequency parts of signals and plotting isolated pulses 
* **xt_maps.py**:
Functions to plot arrays like $E(x,t)$ or $E(x,\omega)$ etc and perform some transforms.

* **xy_maps.py**:
Functions to plot 2D arrays on regular grid.

Example of loading $B_z(x,y)$ field projection and application of spatial filter:
```
import matplotlib.pyplot as plt
from PIC_data_processing.xy_maps import xy_map, SG_filter2D

F = xy_map.from_openPMD(path/to/data, 40000, 'Bz', from_probes = True)
hf_filter = 1 - SG_filter2D(F.shape, F.dx, F.dy, Rmax = 5)# um^-1
F.apply_filter(hf_filter, inplace = True)
F.show(cmap = 'seismic', vmin = -3000, vmax = 3000)
```
* **constants.py**:
Definition of useful constants.
* **math_utils.py**
Contains function to process data: performing fft, fft-filtering of signals, retrieving its envelope and tranforming 2D array from cartesian units to polar
* **plot_utils.py**:
Contains some auxilary functions to plot data
### Scripts description
* **plot_overlap_maps**: takes directorie, that contains output of PIConGPU calculations in form of .h5 data and plots high-frequency part of $B_z$ field projection and electronic density for each .h5-file in each directory. usage: ```python plot_overlapped.py -p path/to/data -t 30000..60000..3000 --fields Ey_Bz```
<p align="center">
<img src="https://github.com/user-attachments/assets/2be0b89c-505a-4253-805d-d6e84fa2f02d" width="400" />
<em align="center"> ROM-harmonic generation, plotted by this scripy (red-$B_z$ field projection, dark blue-electronic density </em>  
</p>

* **process_temporal**: processes spatiotemporal dynamics. e.g.```python process_temporal``` -p path/to/data --to_plot xw_hf``` will plot spectrum for all positions of field observere and high-frequency part of signal.
