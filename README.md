<p align="center">
    <img src="https://github.com/cosmodesi/folpsax/blob/main/folps_logo.png" width="700" height="200">
</p>

# FOLPSν in jax: folpsax

The original code is here: https://github.com/henoriega/FOLPS-nu.
FOLPSν is a Python code that computes the galaxy redshift space power spectrum for cosmologies containing massive neutrinos. The code combines analytical modeling and numerical methods based on the FFTLog formalism. <!-- to speed up the calculations of loop integrals. -->

[![arXiv](https://img.shields.io/badge/arXiv-2208.02791-red)](https://arxiv.org/abs/2208.02791)

## Authors

- [Hernán E. Noriega](mailto:henoriega@estudiantes.fisica.unam.mx)
- [Alejandro Aviles](mailto:avilescervantes@gmail.com)
- Arnaud de Mattia

*Special thanks to other people who have tested this code*: Sebastien Fromenteau, Mariana Vargas-Magaña, Arnaud de Mattia, Gerardo Morales Navarrete


## Requirements

The code employs the standard libraries, and jax:
- numpy
- scipy
- jax
- interpax

## Installation

```
pip install git+https://github.com/cosmodesi/folpsax.git
```

## Run

```python

# First get linear P(k) with cosmoprimo
from cosmoprimo.fiducial import DESI
z = 1.
cosmo = DESI()
fo = cosmo.get_fourier()
pk_interpolator = fo.pk_interpolator().to_1d(z=z)

klin = np.geomspace(1e-5, 10., 1000)
pklin = pk_interpolator(klin)
k0 = 1e-3

# And no-wiggle P(k) (we can also let folpsax compute it, but it's not jaxified yet...)
from cosmoprimo import PowerSpectrumBAOFilter
filter = PowerSpectrumBAOFilter(pk_interpolator, engine='bspline', cosmo=cosmo, cosmo_fid=cosmo)
pknow = filter.smooth_pk_interpolator()(klin)

# Then, folpsax's turn
from folpsax import get_mmatrices, get_non_linear, get_rsd_pkell
mmatrices = get_mmatrices()  # compute matrices once for all

kwargs = {'z': z, 'fnu': cosmo['Omega_ncdm_tot'] / cosmo['Omega_m'], 'Omega_m': cosmo['Omega_m'], 'h': cosmo['h']}
table, table_now = get_non_linear(klin, pklin, mmatrices, pknow=pknow, kminout=0.001, kmaxout=0.5, nk=120, kernels='fk', **kwargs)

k = np.logspace(np.log10(0.01), np.log10(0.3), num=100) # array of  output k in [h/Mpc]
PshotP = 1. / 0.0002118763
# Bias parameters
b1 = 1.645
b2 = -0.46
bs2 = -4./7*(b1 - 1)
b3nl = 32./315*(b1 - 1)
# EFT parameters
alpha0 = 3                 #units: [Mpc/h]^2
alpha2 = -28.9             #units: [Mpc/h]^2
alpha4 = 0.0               #units: [Mpc/h]^2
ctilde = 0.0               #units: [Mpc/h]^4
# Stochatic parameters
alphashot0 = 0.08
alphashot2 = -8.1          #units: [Mpc/h]^2
pars = [b1, b2, bs2, b3nl, alpha0, alpha2, alpha4, ctilde, alphashot0, alphashot2, PshotP]
qpar, qper = 1., 1.

poles = get_rsd_pkell(k, qpar, qper, pars, table, table_now, nmu=6, ells=(0, 2, 4))  # power spectrum multipoles

# We can also jit the theory
import jax

@jax.jit
def compute(dm):
    dk = klin / 0.1
    table, table_now = get_non_linear(klin, pklin * dk**dm, mmatrices, pknow=pknow * dk**dm, kminout=0.001, kmaxout=0.5, nk=120, kernels='fk', **kwargs)
    return get_rsd_pkell(k, qpar, qper, pars, table, table_now, nmu=6, ells=(0, 2, 4))

compute(0.1)  # blazing-fast!

# And autodiff
jax.jacfwd(compute)(0.1)  # derivative w.r.t. dm
```

## TODO

Update docstrings!

Attribution
-----------

Please cite <https://arxiv.org/abs/2208.02791> if you find this code useful in your research.

    @article{Noriega:2022nhf,
    author = "Noriega, Hern\'an E. and Aviles, Alejandro and Fromenteau, Sebastien and Vargas-Maga\~na, Mariana",
    title = "{Fast computation of non-linear power spectrum in cosmologies with massive neutrinos}",
    eprint = "2208.02791",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    month = "8",
    year = "2022"
    }
