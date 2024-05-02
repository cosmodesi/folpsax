#!/usr/bin/env python
# coding: utf-8

#===========================================================================================================================
#============= ============= ============= ============  FOLPSν  ============== ============== ============= ===============
# This is a code for efficiently evaluating the redshift space power spectrum in the presence of massive neutrinos.
#===========================================================================================================================

#standard libraries
from jax import numpy as jnp
from .jax import interp, simpson, legendre
import numpy as np
from scipy import special

from jax import config; config.update('jax_enable_x64', True)


"""
def interp(k, x, y):
    from scipy.interpolate import CubicSpline
    inter = CubicSpline(x, y)
    return inter(k)
"""

def get_mmatrices(nfftlog=128):
    """
    M matrices. They do not depend on the cosmology, so they are computed only once.

    Args:
        if 'nfftlog = None' (or not specified) the code use as default 'nfftlog = 128'.
        to use a different number of sample points, just specify it as 'nfftlog =  number'.
        we recommend using the default mode, see Fig.~8 at arXiv:2208.02791.
    Returns:
        All the M matrices.
    """

    kmin = 10**(-7); kmax = 100.
    b_nu = -0.1;   # not yet tested for other values

    #Eq.~ 4.19 at arXiv:2208.02791
    def Imatrix(nu1, nu2):
        return 1 / (8 * np.pi**(3 / 2.)) * (special.gamma(3 / 2. - nu1) * special.gamma(3 / 2. - nu2) * special.gamma(nu1 + nu2 - 3 / 2.))\
                / (special.gamma(nu1) * special.gamma(nu2) * special.gamma(3 - nu1 - nu2))

    # M22-type
    def M22(nu1, nu2):

        # Overdensity and velocity
        def M22_dd(nu1, nu2):
            return Imatrix(nu1, nu2)*(3/2-nu1-nu2)*(1/2-nu1-nu2)*( (nu1*nu2)*(98*(nu1+nu2)**2 - 14*(nu1+nu2) + 36) - 91*(nu1+nu2)**2+ 3*(nu1+nu2) + 58)/(196*nu1*(1+nu1)*(1/2-nu1)*nu2*(1+nu2)*(1/2-nu2))

        def M22_dt_fp(nu1, nu2):
            return Imatrix(nu1, nu2)*((-3+2*(nu1+nu2))*(-1+2*(nu1+nu2))*(-23-21*nu1+(-38+7*nu1*(-1+7*nu1))*nu2+7*(3+7*nu1)*nu2**2) )/(196*nu1*(1+nu1)*nu2*(1+nu2)*(-1+2*nu2))

        def M22_tt_fpfp(nu1, nu2):
            return Imatrix(nu1, nu2)*((-3+2*(nu1+nu2))*(-12*(1-2*nu2)**2 + 98*nu1**(3)*nu2 + 7*nu1**2*(1+2*nu2*(-8+7*nu2))- nu1*(53+2*nu2*(17+7*nu2))))/(98*nu1*(1+nu1)*nu2*(1+nu2)*(-1+2*nu2))

        def M22_tt_fkmpfp(nu1, nu2):
            return Imatrix(nu1, nu2)*((-3+2*(nu1+nu2))*(-37+7*nu1**(2)*(3+7*nu2) + nu2*(-10+21*nu2) + nu1*(-10+7*nu2*(-1+7*nu2))))/(98*nu1*(1+nu1)*nu2*(1+nu2))

        #A function
        def MtAfp_11(nu1, nu2):
            return Imatrix(nu1, nu2)*((-3+2*(nu1+nu2))*(-1+2*(nu1+nu2))*(-5+nu1*(-4+7*(nu1+nu2))))/(7*nu1*(1+nu1)*(-1+2*nu1)*nu2)

        def MtAfkmpfp_12(nu1, nu2):
            return -Imatrix(nu1, nu2)*(((-3+2*(nu1+nu2))*(-1+2*(nu1+nu2))*(6+7*(nu1+nu2)))/(56*nu1*(1+nu1)*nu2*(1+nu2)))

        def MtAfkmpfp_22(nu1, nu2):
            return Imatrix(nu1, nu2)*((-3+2*(nu1+nu2))*(-18+3*nu1*(1+4*(10-9*nu1)*nu1)+75*nu2+8*nu1*(41+2*nu1*(-28+nu1*(-4+7*nu1)))*nu2+48*nu1*(-9+nu1*(-3+7*nu1))*nu2**2+4*(-39+4*nu1*(-19+35*nu1))*nu2**3+336*nu1*nu2**4) )/(56*nu1*(1+nu1)*(-1+2*nu1)*nu2*(1+nu2)*(-1+2*nu2))

        def MtAfpfp_22(nu1, nu2):
            return Imatrix(nu1, nu2)*((-3+2*(nu1+nu2))*(-5+3*nu2+nu1*(-4+7*(nu1+nu2))))/(7*nu1*(1+nu1)*nu2)

        def MtAfkmpfpfp_23(nu1, nu2):
            return -Imatrix(nu1, nu2)*(((-1+7*nu1)*(-3+2*(nu1+nu2))*(-1+2*(nu1+nu2)))/(28*nu1*(1+nu1)*nu2*(1+nu2)))

        def MtAfkmpfpfp_33(nu1, nu2):
            return Imatrix(nu1, nu2)*((-3+2*(nu1+nu2))*(-1+2*(nu1+nu2))*(-13*(1+nu1)+2*(-11+nu1*(-1+14*nu1))*nu2 + 4*(3+7*nu1)*nu2**2))/(28*nu1*(1+nu1)*nu2*(1+nu2)*(-1+2*nu2))

        #D function
        def MB1_11(nu1, nu2):
            return Imatrix(nu1, nu2)*(3-2*(nu1+nu2))/(4*nu1*nu2)

        def MC1_11(nu1, nu2):
            return Imatrix(nu1, nu2)*((-3+2*nu1)*(-3+2*(nu1+nu2)))/(4*nu2*(1+nu2)*(-1+2*nu2))

        def MB2_11(nu1, nu2):
            return Imatrix(nu1, nu2)*((-3+2*(nu1+nu2))*(-1+2*(nu1+nu2)))/(4*nu1*nu2)

        def MC2_11(nu1, nu2):
            return Imatrix(nu1, nu2)*((-3+2*(nu1+nu2))*(-1+2*(nu1+nu2)))/(4*nu2*(1+nu2))

        def MD2_21(nu1, nu2):
            return Imatrix(nu1, nu2)*((-1+2*nu1-4*nu2)*(-3+2*(nu1+nu2))*(-1+2*(nu1+nu2)))/(4*nu1*nu2*(-1+nu2+2*nu2**2))

        def MD3_21(nu1, nu2):
            return Imatrix(nu1, nu2)*((3-2*(nu1+nu2))*(1-4*(nu1+nu2)**2))/(4*nu1*nu2*(1+nu2))

        def MD2_22(nu1, nu2):
            return Imatrix(nu1, nu2)*(3*(3-2*(nu1+nu2))*(1-2*(nu1+nu2)))/(32*nu1*(1+nu1)*nu2*(1+nu2))

        def MD3_22(nu1, nu2):
            return Imatrix(nu1, nu2)*((3-2*(nu1+nu2))*(1-4*(nu1+nu2)**2)*(1+2*(nu1**2-4*nu1*nu2+nu2**2)))/(16*nu1*(1+nu1)*(-1+2*nu1)*nu2*(1+nu2)*(-1+2*nu2))

        def MD4_22(nu1, nu2):
            return Imatrix(nu1, nu2)*((9-4*(nu1+nu2)**2)*(1-4*(nu1+nu2)**2))/(32*nu1*(1+nu1)*nu2*(1+nu2))


        return (M22_dd(nu1, nu2), M22_dt_fp(nu1, nu2), M22_tt_fpfp(nu1, nu2), M22_tt_fkmpfp(nu1, nu2),
                MtAfp_11(nu1, nu2), MtAfkmpfp_12(nu1, nu2), MtAfkmpfp_22(nu1, nu2), MtAfpfp_22(nu1, nu2),
                MtAfkmpfpfp_23(nu1, nu2), MtAfkmpfpfp_33(nu1, nu2), MB1_11(nu1, nu2), MC1_11(nu1, nu2),
                MB2_11(nu1, nu2), MC2_11(nu1, nu2), MD2_21(nu1, nu2), MD3_21(nu1, nu2), MD2_22(nu1, nu2),
                MD3_22(nu1, nu2), MD4_22(nu1, nu2))


    #M22-type Biasing
    def M22bias(nu1, nu2):

        def MPb1b2(nu1, nu2):
            return Imatrix(nu1, nu2)*((-3+2*(nu1+nu2))*(-4+7*(nu1+nu2)))/(28*nu1*nu2)

        def MPb1bs2(nu1, nu2):
            return Imatrix(nu1, nu2)*((-3+2*(nu1+nu2))*(2+14*nu1**2 *(-1+2*nu2)-nu2*(3+14*nu2)+nu1*(-3+4*nu2*(-11+7*nu2))))/(168*nu1*(1+nu1)*nu2*(1+nu2))

        def MPb22(nu1, nu2):
            return 1/2 * Imatrix(nu1, nu2)

        def MPb2bs2(nu1, nu2):
            return Imatrix(nu1, nu2)*((-3+2*nu1)*(-3+2*nu2))/(12*nu1*nu2)

        def MPb2s2(nu1, nu2):
            return Imatrix(nu1, nu2)*((63-60*nu2+4*(3*(-5+nu1)*nu1+(17-4*nu1)*nu1*nu2+(3+2*(-2+nu1)*nu1)*nu2**2)))/(36*nu1*(1+nu1)*nu2*(1+nu2))

        def MPb2t(nu1, nu2):
            return Imatrix(nu1, nu2)*((-4+7*nu1)*(-3+2*(nu1+nu2)))/(14*nu1*nu2)

        def MPbs2t(nu1, nu2):
            return  Imatrix(nu1, nu2)*((-3+2*(nu1+nu2))*(-19-10*nu2+nu1*(39-30*nu2+14*nu1*(-1+2*nu2))))/(84*nu1*(1+nu1)*nu2*(1+nu2))

        return (MPb1b2(nu1, nu2), MPb1bs2(nu1, nu2), MPb22(nu1, nu2), MPb2bs2(nu1, nu2),
                MPb2s2(nu1, nu2), MPb2t(nu1, nu2), MPbs2t(nu1, nu2))


    #M13-type
    def M13(nu1):

        #Overdensity and velocity
        def M13_dd(nu1):
            return ((1+9*nu1)/4) * np.tan(nu1*np.pi)/(28*np.pi*(nu1+1)*nu1*(nu1-1)*(nu1-2)*(nu1-3) )

        def M13_dt_fk(nu1):
            return ((-7+9*nu1)*np.tan(nu1*np.pi))/(112*np.pi*(-3+nu1)*(-2+nu1)*(-1+nu1)*nu1*(1+nu1))

        def M13_tt_fk(nu1):
            return -(np.tan(nu1*np.pi)/(14*np.pi*(-3 + nu1)*(-2 + nu1)*(-1 + nu1)*nu1*(1 + nu1) ))

        # A function
        def Mafk_11(nu1):
            return ((15-7*nu1)*np.tan(nu1*np.pi))/(56*np.pi*(-3+nu1)*(-2+nu1)*(-1+nu1)*nu1)

        def Mafp_11(nu1):
            return ((-6+7*nu1)*np.tan(nu1*np.pi))/(56*np.pi*(-3+nu1)*(-2+nu1)*(-1+nu1)*nu1)

        def Mafkfp_12(nu1):
            return (3*(-13+7*nu1)*np.tan(nu1*np.pi))/(224*np.pi*(-3+nu1)*(-2+nu1)*(-1+nu1)*nu1*(1+nu1))

        def Mafpfp_12(nu1):
            return (3*(1-7*nu1)*np.tan(nu1*np.pi))/(224*np.pi*(-3+nu1)*(-2+nu1)*(-1+nu1)*nu1*(1+nu1))

        def Mafkfkfp_33(nu1):
            return ((21+(53-28*nu1)*nu1)*np.tan(nu1*np.pi))/(224*np.pi*(-3+nu1)*(-2+nu1)*(-1+nu1)*nu1*(1+nu1))

        def Mafkfpfp_33(nu1):
            return ((-21+nu1*(-17+28*nu1))*np.tan(nu1*np.pi))/(224*np.pi*(-3+nu1)*(-2+nu1)*(-1+nu1)*nu1*(1+nu1))


        return (M13_dd(nu1), M13_dt_fk(nu1), M13_tt_fk(nu1), Mafk_11(nu1),  Mafp_11(nu1), Mafkfp_12(nu1),
                Mafpfp_12(nu1), Mafkfkfp_33(nu1), Mafkfpfp_33(nu1))


    #M13-type Biasing
    def M13bias(nu1):

        def Msigma23(nu1):
            return (45*np.tan(nu1*np.pi))/(128*np.pi*(-3+nu1)*(-2+nu1)*(-1+nu1)*nu1*(1+nu1))

        return (Msigma23(nu1))


    #Computation of M22-type matrices
    def M22type(kmin, kmax, N, b_nu, M22):

        #nuT = -etaT/2, etaT = bias_nu + i*eta_m
        jj = np.arange(N + 1)
        nuT = -0.5 * (b_nu + (2*np.pi*1j/np.log(kmax/kmin)) * (jj - N/2) *(N-1)/(N))

        #reduce time x10 compared to "for" iterations
        nuT_x, nuT_y = np.meshgrid(nuT, nuT)
        M22matrix = M22(nuT_y, nuT_x)

        return np.array(M22matrix)


    #Computation of M13-type matrices
    def M13type(kmin, kmax, N, b_nu, M13):

        #nuT = -etaT/2, etaT = bias_nu + i*eta_m
        ii = np.arange(N + 1)
        nuT = -0.5 * (b_nu + (2*np.pi*1j/np.log(kmax/kmin)) * (ii - N/2) *(N-1)/(N))
        M13vector = M13(nuT)

        return np.array(M13vector)


    #FFTLog bias for the biasing spectra Pb1b2,...
    bnu_b = 15.1*b_nu

    M22T =  M22type(kmin, kmax, nfftlog, b_nu, M22)
    M22biasT = M22type(kmin, kmax, nfftlog, bnu_b, M22bias)
    M22matrices = np.concatenate((M22T, M22biasT))

    M13T = M13type(kmin, kmax, nfftlog, b_nu, M13)
    M13biasT = np.reshape(M13type(kmin, kmax, nfftlog, bnu_b, M13bias), (1, nfftlog + 1))
    M13vectors = np.concatenate((M13T, M13biasT))

    return (M22matrices, M13vectors)


def extrapolate(x, y, xq):
    """
    Extrapolation.

    Args:
        x, y: data set with x- and y-coordinates.
        xq: x-coordinates of extrapolation.
    Returns:
        extrapolates the data set ‘x’, ‘y’  to the range given by ‘xq’.
    """
    def linear_regression(x, y):
        """
        Linear regression.

        Args:
            x, y: data set with x- and y-coordinates.
        Returns:
            slope ‘m’ and the intercept ‘b’.
        """
        xm = jnp.mean(x)
        ym = jnp.mean(y)
        npts = len(x)

        SS_xy = jnp.sum(x * y) - npts * xm * ym
        SS_xx = jnp.sum(x**2) - npts * xm**2
        m = SS_xy / SS_xx

        b = ym - m * xm
        return (m, b)

    m, b = linear_regression(x, y)
    return (xq, m * xq + b)


def extrapolate_pklin(k, pk, extrap=(10**(-7), 200), lim=None):
    """
    Extrapolation to the input linear power spectrum.

    Args:
        k, pk : k-coordinates and linear power spectrum.
    Returns:
        extrapolates the input linear power spectrum ‘pk’ to low-k or high-k if needed.
    """
    k = np.array(k)
    if lim is None: lim = (k[0], k[-1])

    mask = (k >= lim[0]) & (k <= lim[1])
    kcut, pkcut = k[mask], pk[mask]

    kextrap, pkextrap = kcut[:5], pkcut[:5]
    logkinit = np.log10(kextrap[0])
    delta = np.log10(kextrap[1]) - logkinit
    logkextrap = np.arange(logkinit, np.log10(extrap[0]) - delta, -delta)[:0:-1]
    sign_low = jnp.sign(pkextrap[0])
    logk_low, logpk_low = extrapolate(np.log10(np.abs(kextrap)), jnp.log10(jnp.abs(pkextrap)), logkextrap)

    kextrap, pkextrap = kcut[-6:], pkcut[-6:]
    logkinit = np.log10(kextrap[-1])
    delta = logkinit - np.log10(kextrap[-2])
    logkextrap = np.arange(logkinit, np.log10(extrap[1]) + delta, delta)[1:]
    sign_high = jnp.sign(pkextrap[-1])
    logk_high, logpk_high = extrapolate(np.log10(np.abs(kextrap)), jnp.log10(jnp.abs(pkextrap)), logkextrap)

    knew = jnp.concatenate([10**logk_low, kcut, 10**logk_high], axis=0)
    pknew = jnp.concatenate([sign_low * 10**logpk_low, pkcut, sign_high * 10**logpk_high], axis=0)

    return knew, pknew


def get_fnu(h, ombh2, omch2, omnuh2):
    """
    Gives some inputs for the function 'f_over_f0_EH'.

    Args:
        h = H0/100.
        ombh2: Omega_b h² (baryons)
        omch2: Omega_c h² (CDM)
        omnuh2: Omega_nu h² (massive neutrinos)
    Returns:
        h: H0/100.
        OmM0: Omega_b + Omega_c + Omega_nu (dimensionless matter density parameter)
        fnu: Omega_nu/OmM0
        mnu: Total neutrino mass [eV]
    """
    Omb = ombh2 / h**2
    Omc = omch2 / h**2
    Omnu = omnuh2 / h**2

    OmM0 = Omb + Omc + Omnu
    fnu = Omnu / OmM0
    mnu = Omnu * 93.14 * h**2

    return(h, OmM0, fnu, mnu)


def f_over_f0_EH(zev, k, OmM0, h, fnu):
    '''Rutine to get f(k)/f0 and f0.
    f(k)/f0 is obtained following H&E (1998), arXiv:astro-ph/9710216
    f0 is obtained by solving directly the differential equation for the linear growth at large scales.

    Args:
        zev: redshift
        k: wave-number
        OmM0: Omega_b + Omega_c + Omega_nu (dimensionless matter density parameter)
        h = H0/100
        fnu: Omega_nu/OmM0
    Returns:
        f(k)/f0 (when 'EdSkernels = True' f(k)/f0 = 1)
        f0
    '''
    eta = jnp.log(1 / (1 + zev))   #log of scale factor
    Neff = 3.046                   # effective number of neutrinos
    omrv = 2.469*10**(-5)/(h**2 * (1 + 7/8*(4/11)**(4/3)*Neff)) #rad: including neutrinos
    aeq = omrv/OmM0           #matter-radiation equality

    pcb = 5./4 - jnp.sqrt(1 + 24*(1 - fnu))/4     #neutrino supression
    c = 0.7
    Nnu = 3                                     #number of neutrinos
    theta272 = (1.00)**2                        # T_{CMB} = 2.7*(theta272)
    pf = (k * theta272)/(OmM0 * h**2)
    DEdS = jnp.exp(eta)/aeq                      #growth function: EdS cosmology

    yFS = 17.2*fnu*(1 + 0.488*fnu**(-7/6))*(pf*Nnu/fnu)**2  #yFreeStreaming
    rf = DEdS/(1 + yFS)
    fFit = 1 - pcb/(1 + (rf)**c)                #f(k)/f0


    #Getting f0
    def OmM(eta):
        return 1./(1. + ((1-OmM0)/OmM0)*jnp.exp(3*eta) )

    def f1(eta):
        return 2. - 3./2. * OmM(eta)

    def f2(eta):
        return 3./2. * OmM(eta)

    etaini = -6  #initial eta, early enough to evolve as EdS (D + \propto a)
    zfin = -0.99

    def etaofz(z):
        return jnp.log(1/(1 + z))

    etafin = etaofz(zfin)

    from jax.experimental.ode import odeint

    # differential eq.
    def Deqs(Df, eta):
        Df, Dprime = Df
        return [Dprime, f2(eta)*Df - f1(eta)*Dprime]

    # eta range and initial conditions
    eta = jnp.linspace(etaini, etafin, 1001)
    Df0 = jnp.exp(etaini)
    Df_p0 = jnp.exp(etaini)

    # solution
    Dplus, Dplusp = odeint(Deqs, [Df0, Df_p0], eta)
    #print(Dplus, Dplusp)

    Dplusp_ = interp(etaofz(zev), eta, Dplusp)
    Dplus_ = interp(etaofz(zev), eta, Dplus)
    f0 = Dplusp_/Dplus_

    return (k, fFit, f0)


def get_cm(kmin, kmax, N, b_nu, inputpkT):
    """
    Coefficients c_m, see eq.~ 4.2 - 4.5 at arXiv:2208.02791

    Args:
        kmin, kmax: minimal and maximal range of the wave-number k.
        N: number of sampling points (we recommend using N=128).
        b_nu: FFTLog bias (use b_nu = -0.1. Not yet tested for other values).
        inputpkT: k-coordinates and linear power spectrum.
    Returns:
        coefficients c_m (cosmological dependent terms).
    """
    #define de zero matrices
    M = int(N/2)
    k, pk = inputpkT
    ii = jnp.arange(N)

    #"kbins" trought "delta" gives logspaced k's in [kmin, kmax]
    kbins = kmin * jnp.exp(ii * jnp.log(kmax / kmin) / (N - 1))
    f_kl = interp(kbins, k, pk) * (kbins / kmin)**(-b_nu)

    #F_m is the Discrete Fourier Transform (DFT) of f_kl
    #"forward" has the direct transforms scaled by 1/N (numpy version >= 1.20.0)
    F_m = jnp.fft.fft(f_kl, n=N) / N

    #etaT = bias_nu + i*eta_m
    #to get c_m: 1) reality condition, 2) W_m factor
    ii = jnp.arange(N + 1)
    etaT = b_nu + (2*jnp.pi*1j/jnp.log(kmax/kmin)) * (ii - N/2) * (N-1) / N
    c_m = kmin**(-(etaT))*F_m[ii - M]
    c_m = jnp.concatenate([c_m[:1] / 2., c_m[1:-1], c_m[-1:] / 2.])

    return c_m


def get_pknow(k, pk, h):
    """
    Routine (based on J. Hamann et. al. 2010, arXiv:1003.3999) to get the non-wiggle piece of the linear power spectrum.

    Args:
        k: wave-number.
        pk: linear power spectrum.
        h: H0/100.
    Returns:
        non-wiggle piece of the linear power spectrum.
    """
    def interp(k, x, y):  # out-of-range below
        from scipy.interpolate import CubicSpline
        return CubicSpline(x, y)(k)

    from scipy.fft import dst, idst  # not in jax yet...
    #kmin(max): k-range and nk: points
    kmin = 7 * 10**(-5) / h; kmax = 7 / h; nk = 2**16

    #sample ln(kP_L(k)) in nk points, k range (equidistant)
    ksT = kmin + jnp.arange(nk) * (kmax - kmin) / (nk - 1)
    PSL = interp(ksT, k, pk)
    logkpk = jnp.log(ksT * PSL)

    #Discrete sine transf., check documentation
    FSTlogkpkT = dst(np.array(logkpk), type=1, norm="ortho")
    FSTlogkpkOddT = FSTlogkpkT[::2]
    FSTlogkpkEvenT = FSTlogkpkT[1::2]

    #cut range (remove the harmonics around BAO peak)
    mcutmin = 120; mcutmax = 240

    #Even
    xEvenTcutmin = jnp.arange(1, mcutmin-1, 1)
    xEvenTcutmax = jnp.arange(mcutmax+2, len(FSTlogkpkEvenT) + 1, 1)
    EvenTcutmin = FSTlogkpkEvenT[0:mcutmin-2]
    EvenTcutmax = FSTlogkpkEvenT[mcutmax+1:len(FSTlogkpkEvenT)]
    xEvenTcuttedT = jnp.concatenate((xEvenTcutmin, xEvenTcutmax))
    nFSTlogkpkEvenTcuttedT = jnp.concatenate((EvenTcutmin, EvenTcutmax))

    #Odd
    xOddTcutmin = jnp.arange(1, mcutmin, 1)
    xOddTcutmax = jnp.arange(mcutmax+1, len(FSTlogkpkEvenT) + 1, 1)
    OddTcutmin = FSTlogkpkOddT[0:mcutmin-1]
    OddTcutmax = FSTlogkpkOddT[mcutmax:len(FSTlogkpkEvenT)]
    xOddTcuttedT = jnp.concatenate((xOddTcutmin, xOddTcutmax))
    nFSTlogkpkOddTcuttedT = jnp.concatenate((OddTcutmin, OddTcutmax))

    #Interpolate the FST harmonics in the BAO range
    PreEvenT = interp(jnp.arange(2, mcutmax + 1, 1.), xEvenTcuttedT, nFSTlogkpkEvenTcuttedT)
    PreOddT = interp(jnp.arange(0, mcutmax - 1, 1.), xOddTcuttedT, nFSTlogkpkOddTcuttedT)
    preT = jnp.column_stack([PreOddT[mcutmin:mcutmax-1], PreEvenT[mcutmin:mcutmax-1]]).ravel()
    preT = jnp.concatenate([FSTlogkpkT[:2 * mcutmin], preT, FSTlogkpkT[2 * mcutmax - 2:]])

    #Inverse Sine transf.
    FSTofFSTlogkpkNWT = idst(np.array(preT), type=1, norm="ortho")
    PNWT = jnp.exp(FSTofFSTlogkpkNWT)/ksT

    PNWk = interp(k, ksT, PNWT)
    DeltaAppf = k*(PSL[7]-PNWT[7])/PNWT[7]/ksT[7]

    irange1 = k < 1e-3
    PNWk1 = pk[irange1] / (DeltaAppf[irange1] + 1)

    irange2 = (1e-3 <= k) & (k <= ksT[len(ksT)-1])
    PNWk2 = PNWk[irange2]

    irange3 = (k > ksT[len(ksT)-1])
    PNWk3 = pk[irange3]

    PNWkTot = jnp.concatenate([PNWk1, PNWk2, PNWk3])

    return(k, PNWkTot)


def get_non_linear(k, pklin, mmatrices, pknow=None, kminout=0.001, kmaxout=0.5, nk=120, kernels='eds', **kwargs):
    """
    1-loop corrections to the linear power spectrum.

    Args:
        If 'EdSkernels = True' (default: 'False', fk-kernels), EdS-kernels will be employed.
        k, pklin: k-coordinates and linear power spectrum.
        CosmoParams: Set of cosmological parameters [z_pk, omega_b, omega_cdm, omega_ncdm, h] in that order.
                   z_pk: redshift.
                   omega_i = omega_i = Omega_i h², where i=baryons (b), CDM (cdm), massive neutrinos (ncdm).
                   h = H0/100.
    Returns:
        list of 1-loop contributions for the wiggle and non-wiggle (also computed here) linear power spectra.
    """
    kmin = 10**(-7); kmax = 100.
    b_nu = -0.1   #Not yet tested for other values

    #extrapolates the linear power spectrum if needed.
    inputpkT = extrapolate_pklin(k, pklin)

    ################################ KTOUT ###########################################

    #kminout = 0.001; kmaxout = 0.5;

    kTout = jnp.geomspace(kminout, kmaxout, num=nk)

    ##################################################################################

    def P22type(kTout, inputpkT, inputpkTf, inputpkTff, M22matrices, kmin, kmax, N, b_nu):

        (M22_dd, M22_dt_fp, M22_tt_fpfp, M22_tt_fkmpfp, MtAfp_11, MtAfkmpfp_12,
         MtAfkmpfp_22, MtAfpfp_22, MtAfkmpfpfp_23, MtAfkmpfpfp_33, MB1_11, MC1_11,
         MB2_11, MC2_11, MD2_21, MD3_21, MD2_22, MD3_22, MD4_22, MPb1b2, MPb1bs2,
         MPb22, MPb2bs2, MPb2s2, MPb2t, MPbs2t) = M22matrices

        #matter coefficients
        cmT = get_cm(kmin, kmax, N, b_nu, inputpkT)
        cmTf = get_cm(kmin, kmax, N, b_nu, inputpkTf)
        cmTff = get_cm(kmin, kmax, N, b_nu, inputpkTff)

        #biased tracers coefficients
        bnu_b = 15.1 * b_nu
        cmT_b = get_cm(kmin, kmax, N, bnu_b, inputpkT)
        cmTf_b = get_cm(kmin, kmax, N, bnu_b, inputpkTf)

        #etaT = bias_nu + i*eta_m
        jj = jnp.arange(N + 1)
        ietam = (2*jnp.pi*1j/jnp.log(kmax/kmin)) * (jj - N/2) *(N-1) / N
        etamT = b_nu + ietam
        etamT_b = bnu_b + ietam
        K = kTout
        precvec = K[:, None]**(etamT)
        vec = cmT * precvec
        vecf = cmTf * precvec
        vecff = cmTff * precvec

        precvec_b = K[:, None]**(etamT_b)
        vec_b = cmT_b * precvec_b
        vecf_b = cmTf_b * precvec_b

        # Bias
        P22dd = K**3 * jnp.sum(vec @ M22_dd * vec, axis=-1).real
        P22dt = 2*K**3 * jnp.sum(vecf @ M22_dt_fp * vec, axis=-1).real
        P22tt = K**3 * (jnp.sum(vecff @ M22_tt_fpfp * vec, axis=-1) + jnp.sum(vecf @ M22_tt_fkmpfp * vecf, axis=-1)).real

        Pb1b2 = K**3 * jnp.sum(vec_b @ MPb1b2 * vec_b, axis=-1).real
        Pb1bs2 = K**3 * jnp.sum(vec_b @ MPb1bs2 * vec_b, axis=-1).real
        Pb22 = K**3 * jnp.sum(vec_b @ MPb22 * vec_b, axis=-1).real
        Pb2bs2 = K**3 * jnp.sum(vec_b @ MPb2bs2 * vec_b, axis=-1).real
        Pb2s2 = K**3 * jnp.sum(vec_b @ MPb2s2 * vec_b, axis=-1).real
        Pb2t = K**3 * jnp.sum(vecf_b @ MPb2t * vec_b, axis=-1).real
        Pbs2t = K**3 * jnp.sum(vecf_b @ MPbs2t * vec_b, axis=-1).real

        # A-TNS
        I1udd_1b = K**3 * jnp.sum(vecf @ MtAfp_11 * vec, axis=-1).real
        I2uud_1b = K**3 * jnp.sum(vecf @ MtAfkmpfp_12 * vecf, axis=-1).real
        I3uuu_3b = K**3 * jnp.sum(vecff @ MtAfkmpfpfp_33 * vecf, axis=-1).real
        I2uud_2b = K**3 * (jnp.sum(vecf @ MtAfkmpfp_22 * vecf, axis=-1) + jnp.sum(vecff @ MtAfpfp_22 * vec, axis=-1)).real
        I3uuu_2b = K**3 * jnp.sum(vecff @ MtAfkmpfpfp_23 * vecf, axis=-1).real

        # D-RSD
        I2uudd_1D = K**3 * (jnp.sum(vecf @ MB1_11 * vecf, axis=-1) + jnp.sum(vec @ MC1_11 * vecff, axis=-1)).real
        I2uudd_2D = K**3 * (jnp.sum(vecf @ MB2_11 * vecf, axis=-1) + jnp.sum(vec @ MC2_11 * vecff, axis=-1)).real
        I3uuud_2D = K**3 * jnp.sum(vecf @ MD2_21 * vecff, axis=-1).real
        I3uuud_3D = K**3 * jnp.sum(vecf @ MD3_21 * vecff, axis=-1).real
        I4uuuu_2D = K**3 * jnp.sum(vecff @ MD2_22 * vecff, axis=-1).real
        I4uuuu_3D = K**3 * jnp.sum(vecff @ MD3_22 * vecff, axis=-1).real
        I4uuuu_4D = K**3 * jnp.sum(vecff @ MD4_22 * vecff, axis=-1).real

        return (P22dd, P22dt, P22tt, Pb1b2, Pb1bs2, Pb22, Pb2bs2, Pb2s2,
            Pb2t, Pbs2t, I1udd_1b, I2uud_1b, I3uuu_3b, I2uud_2b,
            I3uuu_2b, I2uudd_1D, I2uudd_2D, I3uuud_2D, I3uuud_3D,
            I4uuuu_2D, I4uuuu_3D, I4uuuu_4D)

    def P13type(kTout, inputpkT, inputpkTf, inputpkTff, inputfkT, M13vectors,
                kmin, kmax, N, b_nu):

        (M13_dd, M13_dt_fk, M13_tt_fk, Mafk_11, Mafp_11, Mafkfp_12, Mafpfp_12,
         Mafkfkfp_33, Mafkfpfp_33, Msigma23) = M13vectors

        #condition for EdS-kernels or fk-kernels (default: fk-kernels)
        if kernels == 'eds':
            Fkoverf0 = jnp.ones(len(kTout), dtype='f8')
        else:
            Fkoverf0 = interp(kTout, inputfkT[0], inputfkT[1])

        #matter coefficients
        cmT = get_cm(kmin, kmax, N, b_nu, inputpkT)
        cmTf = get_cm(kmin, kmax, N, b_nu, inputpkTf)
        cmTff = get_cm(kmin, kmax, N, b_nu, inputpkTff)

        #biased tracers coefficients
        bnu_b = 15.1*b_nu
        cmT_b = get_cm(kmin, kmax, N, bnu_b, inputpkT)
        cmTf_b = get_cm(kmin, kmax, N, bnu_b, inputpkTf)

        #etaT = bias_nu + i*eta_m
        jj = jnp.arange(N + 1)
        ietam = (2*jnp.pi*1j/jnp.log(kmax/kmin)) * (jj - N/2) *(N-1) / N
        etamT = b_nu + ietam
        etamT_b = bnu_b + ietam

        sigma2psi = 1/(6 * jnp.pi**2) * simpson(inputpkT[1], x=inputpkT[0])
        sigma2v = 1/(6 * jnp.pi**2) * simpson(inputpkTf[1], x=inputpkTf[0])
        sigma2w = 1/(6 * jnp.pi**2) * simpson(inputpkTff[1], x=inputpkTff[0])
        #print(jnp.sum(cmT), jnp.sum(cmTf), jnp.sum(cmTff), jnp.sum(bnu_b), jnp.sum(cmT_b), jnp.sum(cmTf_b), jnp.sum(sigma2psi), jnp.sum(sigma2v), jnp.sum(sigma2w))

        K = kTout
        precvec = K[:, None]**(etamT)
        vec = cmT * precvec
        vecf = cmTf * precvec
        vecff = cmTff * precvec
        vecfM13dt_fk = vecf @ M13_dt_fk

        precvec_b = K[:, None]**(etamT_b)
        vec_b = cmT_b * precvec_b
        vecf_b = cmTf_b * precvec_b
        # Ploop
        P13dd = K**3 * (vec @ M13_dd).real - 61/105 * K**2 * sigma2psi
        P13dt = 0.5 * K**3 * (Fkoverf0[:, None] * vec @ M13_dt_fk + vecfM13dt_fk).real - (23/21*sigma2psi * Fkoverf0 + 2/21*sigma2v) * K**2
        P13tt = K**3 * (Fkoverf0 * (Fkoverf0[:, None] * vec @ M13_tt_fk + vecfM13dt_fk)).real - (169/105*sigma2psi * Fkoverf0 + 4/21 * sigma2v) * Fkoverf0 * K**2
        # Bias
        sigma23 = K**3 * (vec_b @ Msigma23).real
        # A-TNS
        I1udd_1a = K**3 * (Fkoverf0[:, None] * vec @ Mafk_11 + vecf @ Mafp_11).real + (92/35*sigma2psi * Fkoverf0 - 18/7*sigma2v)*K**2
        I2uud_1a = K**3 * (Fkoverf0[:, None] * vecf @ Mafkfp_12 + vecff @ Mafpfp_12).real - (38/35*Fkoverf0 *sigma2v + 2/7*sigma2w)*K**2
        I3uuu_3a = K**3 * Fkoverf0 * (Fkoverf0[:, None] * vecf @ Mafkfkfp_33 + vecff @ Mafkfpfp_33).real - (16/35*Fkoverf0*sigma2v + 6/7*sigma2w)*Fkoverf0*K**2

        return (P13dd, P13dt, P13tt, sigma23, I1udd_1a, I2uud_1a, I3uuu_3a)


    #Evaluation: f(k)/f0 and linear power spectrum
    #f0 = inputfkT[2]

    #condition for EdS-kernels or fk-kernels (default: fk-kernels)
    if kernels == 'eds':
        inputfkT = None
        f0 = kwargs['f0']
        Fkoverf0 = jnp.ones(len(kTout), dtype='f8')
    else:
        inputfkT = f_over_f0_EH(kwargs['z'], inputpkT[0], kwargs['Omega_m'], kwargs['h'], kwargs['fnu'])
        f0 = kwargs.get('f0', inputfkT[2])
        Fkoverf0 = interp(kTout, inputfkT[0], inputfkT[1])


    #Non-wiggle linear power spectrum
    if pknow is None:
        inputpkT_NW = get_pknow(inputpkT[0], inputpkT[1], kwargs['h'])
    else:
        inputpkT_NW = extrapolate_pklin(k, pknow)

    #condition for EdS-kernels or fk-kernels (default: fk-kernels)
    if kernels == 'eds':
        inputpkTf = (inputpkT[0], inputpkT[1])
        inputpkTff = (inputpkT[0], inputpkT[1])

        inputpkTf_NW = (inputpkT_NW[0], inputpkT_NW[1])
        inputpkTff_NW = (inputpkT_NW[0], inputpkT_NW[1])
    else:
        inputpkTf = (inputpkT[0], inputpkT[1]*inputfkT[1])
        inputpkTff = (inputpkT[0], inputpkT[1]*(inputfkT[1])**2)

        inputpkTf_NW = (inputpkT_NW[0], inputpkT_NW[1]*inputfkT[1])
        inputpkTff_NW = (inputpkT_NW[0], inputpkT_NW[1]*(inputfkT[1])**2)

    M22matrices, M13vectors = mmatrices
    N = M13vectors.shape[-1] - 1
    #P22type contributions
    P22 = P22type(kTout, inputpkT, inputpkTf, inputpkTff, M22matrices, kmin, kmax, N, b_nu)
    P22_NW = P22type(kTout, inputpkT_NW, inputpkTf_NW, inputpkTff_NW, M22matrices, kmin, kmax, N, b_nu)

    #P13type contributions
    P13overpkl = P13type(kTout, inputpkT, inputpkTf, inputpkTff, inputfkT, M13vectors, kmin, kmax, N, b_nu)
    P13overpkl_NW = P13type(kTout, inputpkT_NW, inputpkTf_NW, inputpkTff_NW, inputfkT, M13vectors, kmin, kmax, N, b_nu)
    #print('P22', [float(jnp.mean(a)) for a in P22])
    #print('P13', [float(jnp.mean(a)) for a in P13overpkl])

    #Computations for Table
    pk_l = interp(kTout, inputpkT[0], inputpkT[1])
    pk_l_NW = interp(kTout, inputpkT_NW[0], inputpkT_NW[1])

    sigma2w = 1/(6 * jnp.pi**2) * simpson(inputpkTff[1], x=inputpkTff[0])
    sigma2w_NW = 1/(6 * jnp.pi**2) * simpson(inputpkTff_NW[1], x=inputpkTff_NW[0])

    kbao = 1 / 104
    p = np.geomspace(10**(-6), 0.4, num=100)
    PSL_NW = interp(p, inputpkT_NW[0], inputpkT_NW[1])
    sigma2_NW = 1/(6 * jnp.pi**2) * simpson(PSL_NW * (1 - special.spherical_jn(0, p/kbao) + 2 * special.spherical_jn(2, p/kbao)), x=p)
    delta_sigma2_NW = 1/(2 * jnp.pi**2) * simpson(PSL_NW * special.spherical_jn(2, p/kbao), x=p)

    def remove_zerolag(k, pk):
        # Originally: interp(10**(-10), kTout, P22_NW[5])
        return pk - extrapolate(k[:2], pk[:2], kmin)[1]

    Ploop_dd = P22[0] + P13overpkl[0]*pk_l
    Ploop_dt = P22[1] + P13overpkl[1]*pk_l
    Ploop_tt = P22[2] + P13overpkl[2]*pk_l

    Pb1b2 = P22[3]
    Pb1bs2 = P22[4]
    Pb22 = remove_zerolag(kTout, P22[5])
    Pb2bs2 = remove_zerolag(kTout, P22[6])
    Pb2s2 = remove_zerolag(kTout, P22[7])
    sigma23pkl = P13overpkl[3]*pk_l
    Pb2t = P22[8]
    Pbs2t = P22[9]

    I1udd_1 = P13overpkl[4]*pk_l + P22[10]
    I2uud_1 = P13overpkl[5]*pk_l + P22[11]
    I2uud_2 = (P13overpkl[6]*pk_l)/Fkoverf0 + Fkoverf0*P13overpkl[4]*pk_l + P22[13]
    I3uuu_2 = Fkoverf0*P13overpkl[5]*pk_l + P22[14]
    I3uuu_3 = P13overpkl[6]*pk_l + P22[12]

    I2uudd_1D = P22[15]; I2uudd_2D = P22[16]; I3uuud_2D = P22[17]
    I3uuud_3D = P22[18]; I4uuuu_2D = P22[19]; I4uuuu_3D = P22[20]
    I4uuuu_4D = P22[21]

    TableOut = (kTout, pk_l, Fkoverf0, Ploop_dd, Ploop_dt, Ploop_tt,
                Pb1b2, Pb1bs2, Pb22, Pb2bs2, Pb2s2, sigma23pkl, Pb2t, Pbs2t,
                I1udd_1, I2uud_1, I2uud_2, I3uuu_2, I3uuu_3, I2uudd_1D,
                I2uudd_2D, I3uuud_2D, I3uuud_3D, I4uuuu_2D, I4uuuu_3D,
                I4uuuu_4D, sigma2w, f0)


    ######################## Non- Wiggle ########################################

    Ploop_dd_NW = P22_NW[0] + P13overpkl_NW[0]*pk_l_NW
    Ploop_dt_NW = P22_NW[1] + P13overpkl_NW[1]*pk_l_NW
    Ploop_tt_NW = P22_NW[2] + P13overpkl_NW[2]*pk_l_NW

    Pb1b2_NW = P22_NW[3]
    Pb1bs2_NW = P22_NW[4]
    Pb22_NW = remove_zerolag(kTout, P22_NW[5])
    Pb2bs2_NW = remove_zerolag(kTout, P22_NW[6])
    Pb2s2_NW = remove_zerolag(kTout, P22_NW[7])
    sigma23pkl_NW = P13overpkl_NW[3]*pk_l_NW
    Pb2t_NW = P22_NW[8]
    Pbs2t_NW = P22_NW[9]

    I1udd_1_NW = P13overpkl_NW[4]*pk_l_NW + P22_NW[10]
    I2uud_1_NW = P13overpkl_NW[5]*pk_l_NW + P22_NW[11]
    I2uud_2_NW = (P13overpkl_NW[6]*pk_l_NW)/Fkoverf0 + Fkoverf0*P13overpkl_NW[4]*pk_l_NW + P22_NW[13]
    I3uuu_2_NW = Fkoverf0*P13overpkl_NW[5]*pk_l_NW + P22_NW[14]
    I3uuu_3_NW = P13overpkl_NW[6]*pk_l_NW + P22_NW[12]

    I2uudd_1D_NW = P22_NW[15]; I2uudd_2D_NW = P22_NW[16]; I3uuud_2D_NW = P22_NW[17]
    I3uuud_3D_NW = P22_NW[18]; I4uuuu_2D_NW = P22_NW[19]; I4uuuu_3D_NW = P22_NW[20]
    I4uuuu_4D_NW = P22_NW[21]

    TableOut_NW = (kTout, pk_l_NW, Fkoverf0, Ploop_dd_NW, Ploop_dt_NW, Ploop_tt_NW,
                   Pb1b2_NW, Pb1bs2_NW, Pb22_NW, Pb2bs2_NW, Pb2s2_NW, sigma23pkl_NW,
                   Pb2t_NW, Pbs2t_NW, I1udd_1_NW, I2uud_1_NW, I2uud_2_NW, I3uuu_2_NW,
                   I3uuu_3_NW, I2uudd_1D_NW, I2uudd_2D_NW, I3uuud_2D_NW, I3uuud_3D_NW,
                   I4uuuu_2D_NW, I4uuuu_3D_NW, I4uuuu_4D_NW, sigma2w_NW, sigma2_NW, delta_sigma2_NW, f0)

    return (TableOut, TableOut_NW)


def interp_table(k, table):
    """
    Interpolation of non-linear terms given by the power spectra.

    Args:
        k: wave-number.
    Returns:
        Interpolates the non-linear terms given by the power spectra.
    """
    return tuple(jnp.moveaxis(interp(k, table[0], jnp.column_stack(table[1:26])), -1, 0)) + table[26:]


def get_eft_pkmu(kev, mu, pars, table):
    """
    EFT galaxy power spectrum, Eq. ~ 3.40 at arXiv: 2208.02791.

    Args:
        kev: evaluation points (wave-number coordinates).
        mu: cosine angle between the wave-vector ‘\vec{k}’ and the line-of-sight direction ‘\hat{n}’.
        pars: set of nuisance parameters [b1, b2, bs2, b3nl, alpha0, alpha2, alpha4, ctilde,
                                                  alphashot0, alphashot2, PshotP] in that order.
                    b1, b2, bs2, b3nl: biasing parameters.
                    alpha0, alpha2, alpha4: EFT parameters.
                    ctilde: parameter for NL0 ∝ Kaiser power spectrum.
                    alphashot0, alphashot2, PshotP: stochastic noise parameters.
       table: List of non-linear terms given by the wiggle or non-wiggle power spectra.
    Returns:
       EFT galaxy power spectrum in redshift space.
    """
    # NuisanParams
    (b1, b2, bs2, b3nl, alpha0, alpha2, alpha4, ctilde, alphashot0, alphashot2, PshotP) = pars

    # Table
    (pkl, Fkoverf0, Ploop_dd, Ploop_dt, Ploop_tt, Pb1b2, Pb1bs2, Pb22, Pb2bs2,
    Pb2s2, sigma23pkl, Pb2t, Pbs2t, I1udd_1, I2uud_1, I2uud_2, I3uuu_2, I3uuu_3,
    I2uudd_1D, I2uudd_2D, I3uuud_2D, I3uuud_3D, I4uuuu_2D, I4uuuu_3D, I4uuuu_4D, sigma2w, *_, f0) = table

    fk = Fkoverf0 * f0

    # linear power spectrum
    Pdt_L = pkl*Fkoverf0; Ptt_L = pkl*Fkoverf0**2

    # one-loop power spectrum
    #Pdd = pkl + Ploop_dd; Pdt = Pdt_L + Ploop_dt; Ptt = Ptt_L + Ploop_tt

    # biasing
    def PddXloop(b1, b2, bs2, b3nl):
        return (b1**2 * Ploop_dd + 2*b1*b2*Pb1b2 + 2*b1*bs2*Pb1bs2 + b2**2 * Pb22
                   + 2*b2*bs2*Pb2bs2 + bs2**2 *Pb2s2 + 2*b1*b3nl*sigma23pkl)

    def PdtXloop(b1, b2, bs2, b3nl):
        return b1*Ploop_dt + b2*Pb2t + bs2*Pbs2t + b3nl*Fkoverf0*sigma23pkl

    def PttXloop(b1, b2, bs2, b3nl):
        return Ploop_tt

    # RSD functions
    def Af(mu, f0):
        return (f0*mu**2 * I1udd_1 + f0**2 * (mu**2 * I2uud_1 + mu**4 * I2uud_2)
                    + f0**3 * (mu**4 * I3uuu_2 +  mu**6 * I3uuu_3))

    def Df(mu, f0):
        return (f0**2 * (mu**2 * I2uudd_1D + mu**4 * I2uudd_2D)
                    + f0**3 * (mu**4 * I3uuud_2D + mu**6 * I3uuud_3D)
                    + f0**4 * (mu**4 * I4uuuu_2D + mu**6 * I4uuuu_3D + mu**8 * I4uuuu_4D))

    # Introducing bias in RSD functions, eq.~ A.32 & A.33 at arXiv: 2208.02791
    def ATNS(mu, b1):
        return b1**3 * Af(mu, f0/b1)

    def DRSD(mu, b1):
        return b1**4 * Df(mu, f0/b1)

    def GTNS(mu, b1):
        return -((kev*mu*f0)**2 *sigma2w*(b1**2 * pkl + 2*b1*f0*mu**2 * Pdt_L
                                   + f0**2 * mu**4 * Ptt_L))


    # One-loop SPT power spectrum in redshift space
    def PloopSPTs(mu, b1, b2, bs2, b3nl):
        return (PddXloop(b1, b2, bs2, b3nl) + 2*f0*mu**2 * PdtXloop(b1, b2, bs2, b3nl)
                    + mu**4 * f0**2 * PttXloop(b1, b2, bs2, b3nl) + ATNS(mu, b1) + DRSD(mu, b1)
                    + GTNS(mu, b1))


    # Linear Kaiser power spectrum
    def PKaiserLs(mu, b1):
        return (b1 + mu**2 * fk)**2 * pkl

    def PctNLOs(mu, b1, ctilde):
        return ctilde*(mu*kev*f0)**4 * sigma2w**2 * PKaiserLs(mu, b1)

    # EFT counterterms
    def Pcts(mu, alpha0, alpha2, alpha4):
        return (alpha0 + alpha2 * mu**2 + alpha4 * mu**4) * kev**2 * pkl

    # Stochastics noise
    def Pshot(mu, alphashot0, alphashot2, PshotP):
        return PshotP * (alphashot0 + alphashot2 * (kev*mu)**2)

    return (PloopSPTs(mu, b1, b2, bs2, b3nl) + Pcts(mu, alpha0, alpha2, alpha4)
                + PctNLOs(mu, b1, ctilde) + Pshot(mu, alphashot0, alphashot2, PshotP))


def k_ap(kobs, muobs, qper, qpar):
    r"""
    True ‘k’ coordinates.

    Args: where ‘_obs’ denote quantities that are observed assuming the reference (fiducial) cosmology.
        k_obs: observed wave-number.
        mu_obs: observed cosine angle between the wave-vector ‘\vec{k}’ and the line-of-sight direction ‘\hat{n}’.
        qperp, qpar: AP parameters.
    Returns:
        True wave-number ‘k_AP’.
    """
    F = qpar/qper
    return (kobs/qper)*(1 + muobs**2 * (1./F**2 - 1))**(0.5)


def mu_ap(muobs, qper, qpar):
    r"""
    True ‘mu’ coordinates.

    Args: where ‘_obs’ denote quantities that are observed assuming the reference (fiducial) cosmology.
        mu_obs: observed cosine angle between the wave-vector ‘\vec{k}’ and the line-of-sight direction ‘\hat{n}’.
        qperp, qpar: AP parameters.
    Returns:
        True ‘mu_AP’.
    """
    F = qpar/qper
    return (muobs/F) * (1 + muobs**2 * (1/F**2 - 1))**(-0.5)


def get_rsd_pkmu(k, mu, pars, table, table_now):
    r"""Return redshift space :math:`P(k, \mu)` given input tables."""
    table = interp_table(k, table)
    table_now = interp_table(k, table_now)
    b1 = pars[0]
    f0 = table[-1]
    fk = table[1] * f0
    pkl, pkl_now = table[0], table_now[0]
    sigma2, delta_sigma2 = table_now[-3:-1]
    # Sigma² tot for IR-resummations, see eq.~ 3.59 at arXiv:2208.02791
    sigma2t = (1 + f0*mu**2 * (2 + f0))*sigma2 + (f0*mu)**2 * (mu**2 - 1) * delta_sigma2
    pkmu = ((b1 + fk * mu**2)**2 * (pkl_now + jnp.exp(-k**2 * sigma2t)*(pkl - pkl_now)*(1 + k**2 * sigma2t))
             + jnp.exp(-k**2 * sigma2t) * get_eft_pkmu(k, mu, pars, table)
             + (1 - jnp.exp(-k**2 * sigma2t)) * get_eft_pkmu(k, mu, pars, table_now))
    return pkmu


def get_rsd_pkell(kobs, qpar, qper, pars, table, table_now, nmu=6, ells=(0, 2, 4)):

    r"""Return redshift space :math:`P_{\ell}` given input tables, applying AP transform."""

    def weights_leggauss(nx, sym=False):
        """Return weights for Gauss-Legendre integration."""
        import numpy as np
        x, wx = np.polynomial.legendre.leggauss((1 + sym) * nx)
        if sym:
            x, wx = x[nx:], (wx[nx:] + wx[nx - 1::-1]) / 2.
        return x, wx

    muobs, wmu = weights_leggauss(nmu, sym=True)
    wmu = jnp.array([wmu * (2 * ell + 1) * legendre(ell)(muobs) for ell in ells])
    jac, kap, muap = (qpar * qper**2)**(-3), k_ap(kobs[:, None], muobs, qpar, qper), mu_ap(muobs, qpar, qper)[None, :]
    pkmu = jac * get_rsd_pkmu(kap, muap, pars, table, table_now)
    return jnp.sum(pkmu * wmu[:, None, :], axis=-1)
