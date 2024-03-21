import numpy as np


def test_folps():
    import time
    from cosmoprimo.fiducial import DESI
    from matplotlib import pyplot as plt
    z_pk = 0.5
    k = np.logspace(np.log10(0.01), np.log10(0.3), num=100) # array of k_ev in [h/Mpc]
    PshotP = 1. / 0.0002118763
    # bias parameters
    b1 = 1.645
    b2 = -0.46
    bs2 = -4./7*(b1 - 1)
    b3nl = 32./315*(b1 - 1)
    # EFT parameters
    alpha0 = 3                 #units: [Mpc/h]^2
    alpha2 = -28.9             #units: [Mpc/h]^2
    alpha4 = 0.0               #units: [Mpc/h]^2
    ctilde = 0.0               #units: [Mpc/h]^4
    # Stochatics parameters
    alphashot0 = 0.08
    alphashot2 = -8.1          #units: [Mpc/h]^2
    pars = [b1, b2, bs2, b3nl, alpha0, alpha2, alpha4, ctilde, alphashot0, alphashot2, PshotP]

    for m_ncdm in [0.1, 0.2]:
        cosmo = DESI(m_ncdm=m_ncdm)
        fo = cosmo.get_fourier()
        pk_interpolator = fo.pk_interpolator().to_1d(z=z_pk)
        klin = np.geomspace(1e-5, 10., 1000)
        pklin = pk_interpolator(klin)
        k0 = 1e-3
        f0 = (fo.pk_interpolator(of='theta_cb').to_1d(z=z_pk)(k0) / fo.pk_interpolator(of='delta_cb').to_1d(z=z_pk)(k0))**0.5
        from cosmoprimo import PowerSpectrumBAOFilter
        filter = PowerSpectrumBAOFilter(pk_interpolator, engine='bspline', cosmo=cosmo, cosmo_fid=cosmo)
        pknow = filter.smooth_pk_interpolator()(klin)

        omega_b, omega_cdm, omega_ncdm, h = cosmo['omega_b'], cosmo['omega_cdm'], cosmo['omega_ncdm_tot'], cosmo['h']
        CosmoParams = [z_pk, omega_b, omega_cdm, omega_ncdm, h]
        inputpkT = [klin, pklin]

        import FOLPSnu as FOLPS
        matrices = FOLPS.Matrices()
        t0 = time.time()
        nonlinear = FOLPS.NonLinear(inputpkT, CosmoParams, EdSkernels=False)
        ref = FOLPS.RSDmultipoles(k, pars, AP=False)[1:]
        print(time.time() - t0)
        #print(f0 / FOLPS.f0, f0, FOLPS.f0)
        print('=' * 20)
        from folpsax import get_mmatrices, get_non_linear, get_rsd_pkell
        mmatrices = get_mmatrices()
        kwargs = {'z': z_pk, 'fnu': cosmo['Omega_ncdm_tot'] / cosmo['Omega_m'], 'Omega_m': cosmo['Omega_m'], 'h': cosmo['h']}
        t0 = time.time()
        table, table_now = get_non_linear(klin, pklin, mmatrices, pknow=pknow, kminout=0.001, kmaxout=0.5, nk=120, kernels='fk', **kwargs)
        print([float(np.mean(a)) for a in FOLPS.TableOut_NW])
        print([float(np.mean(a)) for a in table_now])
        #print(table[-2] / FOLPS.f0)
        qpar, qper = 1., 1.
        poles = get_rsd_pkell(k, qpar, qper, pars, table, table_now, nmu=6, ells=(0, 2, 4))
        print(time.time() - t0)

        ax = plt.gca()
        for ill, ell in enumerate((0, 2, 4)):
            ax.plot(k, k * ref[ill], color='C{:d}'.format(ill), ls='--')
            ax.plot(k, k * poles[ill], color='C{:d}'.format(ill), ls='-', label=r'$\ell = {:d}$'.format(ell))
        ax.set_xlim([k[0], k[-1]])
        ax.grid(True)
        ax.legend()
        ax.set_ylabel(r'$k \Delta P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{2}$]')
        ax.set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
        plt.show()


def test_autodiff():
    import time
    from cosmoprimo.fiducial import DESI
    z = 0.5
    k = np.logspace(np.log10(0.01), np.log10(0.3), num=100) # array of k_ev in [h/Mpc]
    PshotP = 1. / 0.0002118763
    # bias parameters
    b1 = 1.645
    b2 = -0.46
    bs2 = -4./7*(b1 - 1)
    b3nl = 32./315*(b1 - 1)
    # EFT parameters
    alpha0 = 3                 #units: [Mpc/h]^2
    alpha2 = -28.9             #units: [Mpc/h]^2
    alpha4 = 0.0               #units: [Mpc/h]^2
    ctilde = 0.0               #units: [Mpc/h]^4
    # Stochatics parameters
    alphashot0 = 0.08
    alphashot2 = -8.1          #units: [Mpc/h]^2
    pars = [b1, b2, bs2, b3nl, alpha0, alpha2, alpha4, ctilde, alphashot0, alphashot2, PshotP]
    qpar, qper = 1., 1.

    cosmo = DESI()
    fo = cosmo.get_fourier()
    pk_interpolator = fo.pk_interpolator().to_1d(z=z)
    klin = np.geomspace(1e-5, 10., 1000)
    pklin = pk_interpolator(klin)
    k0 = 1e-3
    f0 = (fo.pk_interpolator(of='theta_cb').to_1d(z=z)(k0) / fo.pk_interpolator(of='delta_cb').to_1d(z=z)(k0))**0.5
    from cosmoprimo import PowerSpectrumBAOFilter
    filter = PowerSpectrumBAOFilter(pk_interpolator, engine='bspline', cosmo=cosmo, cosmo_fid=cosmo)
    pknow = filter.smooth_pk_interpolator()(klin)

    from folpsax import get_mmatrices, get_non_linear, get_rsd_pkell
    mmatrices = get_mmatrices()
    kwargs = {'z': z, 'fnu': cosmo['Omega_ncdm_tot'] / cosmo['Omega_m'], 'Omega_m': cosmo['Omega_m'], 'h': cosmo['h']}

    import jax

    def compute(pklin, pknow):
        table, table_now = get_non_linear(klin, pklin, mmatrices, pknow=pknow, kminout=0.001, kmaxout=0.5, nk=120, kernels='fk', **kwargs)
        return get_rsd_pkell(k, qpar, qper, pars, table, table_now, nmu=6, ells=(0, 2, 4))

    dk = klin / 0.1
    def get_pk(dm):
        return pklin * dk**dm, pknow * dk**dm

    pk1 = compute(*get_pk(0.2))
    compute2 = jax.jit(compute)
    compute2(*get_pk(0.1))
    t0 = time.time()
    niterations = 5
    for ii in range(niterations): compute2(*get_pk(0.1 + 0.01 * ii))
    print((time.time() - t0) / niterations)
    pk2 = compute2(*get_pk(0.2))

    @jax.jit
    def compute3(dm):
        pklin, pknow = get_pk(dm)
        return compute2(pklin, pknow)

    compute3(0.1)
    t0 = time.time()
    niterations = 5
    for ii in range(niterations): compute3(0.1 + 0.01 * ii)
    print((time.time() - t0) / niterations)

    from matplotlib import pyplot as plt
    ax = plt.gca()
    for ill, ell in enumerate((0, 2, 4)):
        ax.plot(k, k * pk1[ill], color='C{:d}'.format(ill), ls='--')
        ax.plot(k, k * pk2[ill], color='C{:d}'.format(ill), ls='-', label=r'$\ell = {:d}$'.format(ell))
    ax.set_xlim([k[0], k[-1]])
    ax.grid(True)
    ax.legend()
    ax.set_ylabel(r'$k \Delta P_{\ell}(k)$ [$(\mathrm{Mpc}/h)^{2}$]')
    ax.set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')
    plt.show()

    def compute3(dm):
        pklin, pknow = get_pk(dm)
        table, table_now = get_non_linear(klin, pklin, mmatrices, pknow=pknow, kminout=0.001, kmaxout=0.5, nk=120, kernels='fk', **kwargs)
        return get_rsd_pkell(k, qpar, qper, pars, table, table_now, nmu=6, ells=(0, 2, 4))

    compute3 = jax.jit(compute3)
    print(jax.jacfwd(compute3)(0.1))


def test_nowiggle():

    from cosmoprimo.fiducial import DESI
    z_pk = 0.5
    cosmo = DESI()
    fo = cosmo.get_fourier()
    pk_interpolator = fo.pk_interpolator().to_1d(z=z_pk)
    klin = np.geomspace(1e-5, 10., 1000)
    pklin = pk_interpolator(klin)

    h = cosmo['h']
    from folpsax.folps import extrapolate_pklin, get_pknow
    from FOLPSnu import pknwJ

    klin, pklin = extrapolate_pklin(klin, pklin)
    pkref = pknwJ(klin, pklin, h)[1]
    pknow = get_pknow(klin, pklin, h)[1]

    print(pknow / pkref)
    from matplotlib import pyplot as plt
    ax = plt.gca()
    ax.plot(klin, pklin / pkref)
    ax.plot(klin, pklin / pknow)
    ax.set_xlim(0., 0.3)
    plt.show()



if __name__ == '__main__':

    test_folps()
    test_autodiff()
    #test_nowiggle()