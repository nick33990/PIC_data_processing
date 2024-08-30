from math import pi

# Some useful constants in SI units & conversion factors
c = 299792458
k = 1.380649e-23 
h = 6.62607015e-34
e = 1.6e-19
me = 9.1e-31
h_ = h / (2 * pi)
eps0 = 8.85e-12
mu0 = 1.25e-6
E2I = 0.5 * eps0 * c * 1e-4 # conversion of electric field (in V/m) squared to intensity (in W/cm^2), I = E2I * E ** 2
B2I = 0.5 / mu0 * c * 1e-4 # conversion of magnetic field (in Tl) squared to intensity (in W/cm^2), I = B2I * B ** 2
w2E = h_ / e # conversion of angular frquency (in rad/s) to photon energy (in eV)
wl2nc = 1.142e27 # conversion of inverse squared laser wavelenght (in um) to corresponding critial density (1/m^3) nc = wl2nc / wavelenght^2