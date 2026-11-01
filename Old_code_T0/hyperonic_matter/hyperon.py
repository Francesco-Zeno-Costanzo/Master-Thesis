"""
This code computes the energy per nucleon in asymmetric matter
with the sigma-omega model in relativistic mean field at T = 0
considering beta equilibrium between protons, electrons, neutrons,
muons and lambdaa
"""

import time
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from scipy.optimize import root

#\hbar/c in MeV fermi
hbc = 197.327

#saturation numerical densitity
nb_sat = 0.152

#baryon masses
mass_p  = 938.27203
mass_n  = 939.56536
mass_L  = 1115.683
mass_Sp = 1189.37
mass_S0 = 1192.642
mass_Sm = 1197.45
mass_X0 = 1314.86
mass_Xm = 1321.71
m_n = np.array([mass_p,  mass_n,  mass_L,  mass_Sp,
                mass_S0, mass_Sm, mass_X0, mass_Xm])/hbc

#meson masses
m_sig = 550.1238
m_omg = 783.0000
m_rho = 763.0000
m_sgs = 980.0
m_phi = 1020
m_m = np.array([m_sig, m_omg, m_rho, m_sgs, m_phi])/hbc

#lepton masses
m_el = 0.51099907
m_mu = 105.6583755
m_l = np.array([m_el, m_mu])/hbc

#coupling constants at nb_sat
g_sg_sat = 10.5396
g_om_sat = 13.0189
g_rh_sat = 3.6836 #we already considered 1/2
g_sat_NN = np.array([g_sg_sat, g_om_sat, g_rh_sat])

#hyperonic couplig costant Lambda-meson
x_sig_L0 = 0.614
x_omg_L0 = 2/3
x_phi_L0 = - np.sqrt(2)/3
x_sgs_L0 = 0.538

g_sig_L_sat = x_sig_L0 * g_sg_sat
g_omg_L_sat = x_omg_L0 * g_om_sat
g_phi_L_sat = x_phi_L0 * g_om_sat
g_sgs_L_sat = x_sgs_L0 * g_sg_sat
#in order            sigma,       omega,       phi           sigma*
g_sat_L0 = np.array([g_sig_L_sat, g_omg_L_sat, g_phi_L_sat, g_sgs_L_sat])

#hyperonic couplig costant Sigma_p-meson
x_sig_S = 0.5353831#0.460760
x_omg_S = 2/3
x_rho_S = 2

g_sig_Sp_sat = x_sig_S * g_sg_sat
g_omg_Sp_sat = x_omg_S * g_om_sat
g_rho_Sp_sat = x_rho_S * g_rh_sat
g_sat_Sp = np.array([g_sig_Sp_sat, g_omg_Sp_sat, g_rho_Sp_sat])

#hyperonic couplig costant Sigma_0-meson
g_sig_S0_sat = x_sig_S * g_sg_sat
g_omg_S0_sat = x_omg_S * g_om_sat
g_sat_S0 = np.array([g_sig_S0_sat, g_omg_S0_sat])

#hyperonic couplig costant Sigma_m-meson
g_sig_Sm_sat = x_sig_S * g_sg_sat
g_omg_Sm_sat = x_omg_S * g_om_sat
g_rho_Sm_sat = x_rho_S * g_rh_sat
g_sat_Sm = np.array([g_sig_Sm_sat, g_omg_Sm_sat, g_rho_Sm_sat])

#hyperonic couplig costant Xi_0-meson
x_sig_X = 0.30251540
x_omg_X = 1/3
x_rho_X = 1

g_sig_X0_sat = x_sig_X * g_sg_sat
g_omg_X0_sat = x_omg_X * g_om_sat
g_rho_X0_sat = x_rho_X * g_rh_sat
g_sat_X0 = np.array([g_sig_X0_sat, g_omg_X0_sat, g_rho_X0_sat])

#hyperonic couplig costant Xi_m-meson
g_sig_Xm_sat = x_sig_X * g_sg_sat
g_omg_Xm_sat = x_omg_X * g_om_sat
g_rho_Xm_sat = x_rho_X * g_rh_sat
g_sat_Xm = np.array([g_sig_Xm_sat, g_omg_Xm_sat, g_rho_Xm_sat])

#array to model the dependence on the coupling constants
#in order     sigma,  omega,  rho
a = np.array([1.3881, 1.3892, 0.5647])
b = np.array([1.0943, 0.9240])
c = np.array([1.7057, 1.4620])
d = np.array([0.4421, 0.4775])


def f(n, i, derive=False):
    """
    function that models the behaviour of the
    coupling constants depending on nb
    for the sigma meson and the omega meson
    It returns the derivative depending on the value of derive

    Parameters
    ----------
    n : float or 1darray
        total baryon density
    i : int
        it is a flag
        i = 0 sigma
        i = 1 omega
    derive : boolen
        it is a flag if True returns the derivative

    Return
    ----------
    derive=False
        g_n : float or 1darray
            g(nb/nb_sat)
    derive=True
        dg_n :float or 1d array
            dg/dnb (nb/nb_sat)
    """

    nb_sat = 0.152
    x = n/nb_sat

    if not derive :
        num = 1 + b[i]*(x + d[i])**2
        den = 1 + c[i]*(x + d[i])**2
        g_n = a[i] * num/den
        return g_n

    else :
        num = 2*a[i]*(b[i] - c[i])*nb_sat**2 * (x + d[i]*nb_sat)
        den = (nb_sat**2 + c[i]*(x + d[i]*nb_sat)**2)**2
        dg_n = num/den
        return dg_n


def g(n, derive=False):
    """
    function that models the behaviour of the
    coupling constant depending on nb
    for the rho meson

    Parameters
    ----------
    x : float or 1darray
        nb/nb_sat
    derive : boolen
        it is a flag if True returns the derivative

    Returns
    ----------
    derive=False
        g_n : float or 1darray
            g(nb/nb_sat)
    derive=True
        dg_n :float or 1d array
            dg/dnb (nb/nb_sat)
    """

    nb_sat = 0.152
    x = n/nb_sat

    if not derive :
        g_n = np.exp(-a[2]*(x-1))
        return g_n

    else :
        dg_n = - a[2]/nb_sat * np.exp(a[2]*(1-x))
        return dg_n


def ns(k, M):
    """
    integral for the sigma equation

    Parameters
    ----------
    k : float
        momentum, integration variable
    M : float
        effective mass

    Return
    ----------
    x_dot : float
        integrand function
    """

    num = k**2 * M
    den = np.sqrt(k**2 + M**2)

    x_dot = 1/(np.pi**2) * num/den

    return x_dot


def ene(k, M):
    """
    integral for the nucleons energy

    Parameters
    ----------
    k : float
        momentum, integration variable
    M : float
        mass or effective mass

    Return
    ----------
    x_dot : float
        integrand function
    """
    eps = k**2 * np.sqrt(k**2 + M**2)
    x_dot = eps/(np.pi**2)

    return x_dot


def pre(k, M):
    """
    integral for the nucleons pressure

    Parameters
    ----------
    k : float
        momentum, integration variable
    M : float
        mass or effective mass

    Return
    ----------
    x_dot : float
        integrand function
    """
    eps = k**4 / np.sqrt(k**2 + M**2)
    x_dot = eps/(np.pi**2)

    return x_dot


def Inte(f, kf, M):
    """
    this function computes the integral via scipy

    Parameters
    ----------
    f : function
        integrand function
    kf : float
        fermi momentum, integral upper bound
    M : float
        mass

    Return
    -----------
    I : float
        \int_0^kf f(k, M) dk
    """
    I, dI = integrate.quad(f, 0, kf, args=(M,))

    return I


def system(V, nb,
            g_sig_NN, g_omg_NN, g_rho_NN,
            g_sig_L0, g_omg_L0, g_phi_L0, g_sgs_L0,
            g_sig_Sp, g_omg_Sp, g_rho_Sp,
            g_sig_S0, g_omg_S0,
            g_sig_Sm, g_omg_Sm, g_rho_Sm,
            g_sig_X0, g_omg_X0, g_rho_X0,
            g_sig_Xm, g_omg_Xm, g_rho_Xm,
            muon, Lambda0, Sigma_p, Sigma_0, Sigma_m, Xi_0, Xi_m):
    """
    system to solve

    Parameters
    ----------
    V : 1darray
        array of the variables
    nb : float
        total baryonic density

    g_sig_NN, ..., g_rho_Xm : float
        several coupling*constant

    muon, ..., Xi_m : bool
        flag for presence of particles


    Returns
    ----------
    list of equations to solve
    ri = LHS - RHS = 0
    if ri = 0 alredy satisfied
    """
    sig, omg, rho, sgs, phi, n, p, e, m, L0, Sp, S0, Sm, X0, Xm = V

    n = abs(n)         # to avoid smal negative value
    p = abs(p)         # that can be problematic in
    e = abs(e)         # the power for kf_i
    m = abs(m)         # 1e-9 and -1e-9 are practically
    L0 = abs(L0)       # zero but with the second the
    Sp = abs(Sp)       # the elevation to power in kf_i
    S0 = abs(S0)       # is not define
    Sm = abs(Sm)
    X0 = abs(X0)
    Xm = abs(Xm)

    #fermi momenta
    kf_p = (3*(np.pi**2)*p)**(1/3)
    kf_n = (3*(np.pi**2)*n)**(1/3)
    kf_e = (3*(np.pi**2)*e)**(1/3)
    #effective masses
    m_eff_p  = m_n[0] - sig*g_sig_NN
    m_eff_n  = m_n[1] - sig*g_sig_NN
    m_eff_L0 = m_n[2] - sig*g_sig_L0 - sgs*g_sgs_L0
    m_eff_Sp = m_n[3] - sig*g_sig_Sp
    m_eff_S0 = m_n[4] - sig*g_sig_S0
    m_eff_Sm = m_n[5] - sig*g_sig_Sm
    m_eff_X0 = m_n[6] - sig*g_sig_X0
    m_eff_Xm = m_n[7] - sig*g_sig_Xm
    #integrals for scalar density
    I_p = Inte(ns, kf_p, m_eff_p)
    I_n = Inte(ns, kf_n, m_eff_n)
    #chemical potentials for baryons and electrons
    mu_p = np.sqrt(kf_p**2 + m_eff_p**2) - g_rho_NN*rho + g_omg_NN*omg
    mu_n = np.sqrt(kf_n**2 + m_eff_n**2) + g_rho_NN*rho + g_omg_NN*omg
    mu_e = np.sqrt(kf_e**2 + m_l[0]**2)

    I_sg_n = g_sig_NN*(I_p + I_n)  #all integrals for sigma
    I_sg_s = 0                     #all integrals for sigma*
    ND_omg = g_omg_NN*(n + p)      #density for omega
    ND_rho = g_rho_NN*(n - p)      #density for rho
    ND_phi = 0                     #density for phi
    N_TOT = n + p                  #Total density
    TPC = p                        #Total positive charge
    TNC = e                        #Total negative charge


    #equations to solve
    if muon:
        #muons
        kf_m = (3*(np.pi**2)*m)**(1/3)
        mu_m = np.sqrt(kf_m**2 + m_l[1]**2)

        TNC += m

    if Lambda0:
        #lambdas
        kf_L0 = (3*(np.pi**2)*L0)**(1/3)
        mu_L0 = np.sqrt(kf_L0**2 + m_eff_L0**2) + g_phi_L0*phi + g_omg_L0*omg
        I_L0  = Inte(ns, kf_L0, m_eff_L0)

        I_sg_n += g_sig_L0 * I_L0
        I_sg_s += g_sgs_L0 * I_L0
        ND_omg += g_omg_L0 * L0
        ND_phi += g_phi_L0 * L0
        N_TOT  += L0

    if Sigma_p:
        #Sigma_p
        kf_Sp = (3*(np.pi**2)*Sp)**(1/3)
        mu_Sp = np.sqrt(kf_Sp**2 + m_eff_Sp**2) - g_rho_Sp*rho + g_omg_Sp*omg
        I_Sp  = Inte(ns, kf_Sp, m_eff_Sp)

        I_sg_n +=  g_sig_Sp * I_Sp
        ND_omg +=  g_omg_Sp * Sp
        ND_rho += -g_rho_Sp * Sp
        N_TOT  += Sp
        TPC    += Sp

    if Sigma_0:
        #Sigma_0
        kf_S0 = (3*(np.pi**2)*S0)**(1/3)
        mu_S0 = np.sqrt(kf_S0**2 + m_eff_S0**2) + g_omg_S0*omg
        I_S0  = Inte(ns, kf_S0, m_eff_S0)

        I_sg_n += g_sig_S0 * I_S0
        ND_omg += g_omg_S0 * S0
        N_TOT  += S0

    if Sigma_m:
        #Sigma_m
        kf_Sm = (3*(np.pi**2)*Sm)**(1/3)
        mu_Sm = np.sqrt(kf_Sm**2 + m_eff_Sm**2) + g_rho_Sm*rho + g_omg_Sm*omg
        I_Sm  = Inte(ns, kf_Sm, m_eff_Sm)

        I_sg_s += g_sig_Sm * I_Sm
        ND_omg += g_omg_Sm * Sm
        ND_rho += g_rho_Sm * Sm
        N_TOT  += Sm
        TNC    += Sm

    if Xi_0:
        #Xi_0
        kf_X0 = (3*(np.pi**2)*X0)**(1/3)
        mu_X0 = np.sqrt(kf_X0**2 + m_eff_X0**2) - g_rho_X0*rho + g_omg_X0*omg
        I_X0  = Inte(ns, kf_X0, m_eff_X0)

        I_sg_n +=  g_sig_X0 * I_X0
        ND_omg +=  g_omg_X0 * X0
        ND_rho += -g_rho_X0 * X0
        N_TOT  += X0

    if Xi_m:
        #Xi_m
        kf_Xm = (3*(np.pi**2)*Xm)**(1/3)
        mu_Xm = np.sqrt(kf_Xm**2 + m_eff_Xm**2) + g_rho_Xm*rho + g_omg_Xm*omg
        I_Xm  = Inte(ns, kf_Xm, m_eff_Xm)

        I_sg_n += g_sig_Xm * I_Xm
        ND_omg += g_omg_Xm * Xm
        ND_rho += g_rho_Xm * Xm
        N_TOT  += Xm
        TNC    += Xm

    #eq for mesons
    r1 = sig - 1/m_m[0]**2 * I_sg_n
    r2 = omg - 1/m_m[1]**2 * ND_omg
    r3 = rho - 1/m_m[2]**2 * ND_rho
    r4 = sgs - 1/m_m[3]**2 * I_sg_s
    r5 = phi - 1/m_m[4]**2 * ND_phi
    #conservation of barionic number
    r6 = nb - N_TOT
    #conservation of charge
    r7 = TPC - TNC
    #eq chemical potential
    r8  = mu_n - mu_p - mu_e
    r9  = mu_e - mu_m           if muon    else 0
    r10 = mu_n - mu_L0          if Lambda0 else 0
    r11 = mu_p - mu_Sp          if Sigma_p else 0
    r12 = mu_n - mu_S0          if Sigma_0 else 0
    r13 = mu_n + mu_e - mu_Sm   if Sigma_m else 0
    r14 = mu_n - mu_X0          if Xi_0    else 0
    r15 = mu_n + mu_e - mu_Xm   if Xi_m    else 0


    return [r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r1, r12, r13, r14, r15]


def total_energy(nb_dens,
                 n_pro, n_neu, n_lam, n_Sgp, n_Sg0, n_Sgm, n_Xi0,
                 n_Xim, n_ele, n_muo, sigma, omega, rho, sig_s, phi):
    """
    Computation of the system's total energy

    Parameters
    ----------
    nb_dens : 1darray
        total baryonic density

    n_pro, ..., n_muo : 1darrray
        array for the particle densities
        If a certain particle isn't present
        the density will be zeo, so also the
        fermi momentum will be zero and so the
        relative integral

    sigma, omega, rho, sig_s, phi : 1darray
        array for the mesons solutions

    Retunrs
    ----------
    energ : 1darray
        array for the total energy
    """
    number_dens = len(nb_dens)
    #array for the energy
    energ = np.zeros(number_dens)

    for i, nb in enumerate(nb_dens):
        #fermi momenta
        kf_n  = (3*(np.pi**2)*n_neu[i])**(1/3)
        kf_p  = (3*(np.pi**2)*n_pro[i])**(1/3)
        kf_L0 = (3*(np.pi**2)*n_lam[i])**(1/3)
        kf_Sp = (3*(np.pi**2)*n_Sgp[i])**(1/3)
        kf_S0 = (3*(np.pi**2)*n_Sg0[i])**(1/3)
        kf_Sm = (3*(np.pi**2)*n_Sgm[i])**(1/3)
        kf_X0 = (3*(np.pi**2)*n_Xi0[i])**(1/3)
        kf_Xm = (3*(np.pi**2)*n_Xim[i])**(1/3)
        kf_m  = (3*(np.pi**2)*n_muo[i])**(1/3)
        kf_e  = (3*(np.pi**2)*n_ele[i])**(1/3)
        #constants as the density varies for baryons
        g_sig_NN = g_sat_NN[0] * f(nb, 0)
        g_sig_L0 = g_sat_L0[0]# * f(nb, 0)
        g_sgs_L0 = g_sat_L0[3]# * f(nb, 0)
        g_sig_Sp = g_sat_Sp[0]# * f(nb, 0)
        g_sig_S0 = g_sat_S0[0]# * f(nb, 0)
        g_sig_Sm = g_sat_Sm[0]# * f(nb, 0)
        g_sig_X0 = g_sat_X0[0]# * f(nb, 0)
        g_sig_Xm = g_sat_Xm[0]# * f(nb, 0)
        #effective masses
        m_eff_p  = m_n[0] - sigma[i]*g_sig_NN
        m_eff_n  = m_n[1] - sigma[i]*g_sig_NN
        m_eff_L0 = m_n[2] - sigma[i]*g_sig_L0 - sig_s[i]*g_sgs_L0
        m_eff_Sp = m_n[3] - sigma[i]*g_sig_Sp
        m_eff_S0 = m_n[4] - sigma[i]*g_sig_S0
        m_eff_Sm = m_n[5] - sigma[i]*g_sig_Sm
        m_eff_X0 = m_n[6] - sigma[i]*g_sig_X0
        m_eff_Xm = m_n[7] - sigma[i]*g_sig_Xm
        #mesons energy
        ene_m  = 0.5*(m_m[0]**2*sigma[i]**2 + m_m[1]**2*omega[i]**2 + m_m[2]**2*rho[i]**2)
        ene_m += 0.5*(m_m[3]**2*sig_s[i]**2 + m_m[4]**2*phi[i]**2)
        #nucleons energies
        ene_p  = Inte(ene, kf_p,  m_eff_p)
        ene_n  = Inte(ene, kf_n,  m_eff_n)
        ene_L0 = Inte(ene, kf_L0, m_eff_L0)
        ene_Sp = Inte(ene, kf_Sp, m_eff_Sp)
        ene_S0 = Inte(ene, kf_S0, m_eff_S0)
        ene_Sm = Inte(ene, kf_Sm, m_eff_Sm)
        ene_X0 = Inte(ene, kf_X0, m_eff_X0)
        ene_Xm = Inte(ene, kf_Xm, m_eff_Xm)
        ene_BT = ene_p + ene_n + ene_L0 + ene_Sp + ene_S0 + ene_Sm + ene_X0 + ene_Xm
        #leptons energy term
        ene_e  = Inte(ene, kf_e, m_l[0])
        ene_mu = Inte(ene, kf_m, m_l[1])
        #system's total energy
        energ[i] =  ene_m + ene_BT + ene_e + ene_mu

    return energ


def total_pressure(nb_dens,
                   n_pro, n_neu, n_lam, n_Sgp, n_Sg0, n_Sgm, n_Xi0,
                   n_Xim, n_ele, n_muo, sigma, omega, rho, sig_s, phi):
    """
    Computation of the system's total pressure

    Parameters
    ----------
    nb_dens : 1darray
        total baryonic density

    n_pro, ...,n_muo : 1darrray
        array for the particle densities
        If a certain particle isn't present
        the density will be zeo, so also the
        fermi momentum will be zero and so the
        relative integral

    sigma, omega, rho, sig_s, phi : 1darray
        array for the mesons solutions

    Retunrs
    ----------
    press : 1darray
        array for the total pressure
    """
    number_dens = len(nb_dens)
    #array for the pressure
    press = np.zeros(number_dens)

    for i, nb in enumerate(nb_dens):
        #fermi momenta
        kf_n  = (3*(np.pi**2)*n_neu[i])**(1/3)
        kf_p  = (3*(np.pi**2)*n_pro[i])**(1/3)
        kf_L0 = (3*(np.pi**2)*n_lam[i])**(1/3)
        kf_Sp = (3*(np.pi**2)*n_Sgp[i])**(1/3)
        kf_S0 = (3*(np.pi**2)*n_Sg0[i])**(1/3)
        kf_Sm = (3*(np.pi**2)*n_Sgm[i])**(1/3)
        kf_X0 = (3*(np.pi**2)*n_Xi0[i])**(1/3)
        kf_Xm = (3*(np.pi**2)*n_Xim[i])**(1/3)
        kf_m  = (3*(np.pi**2)*n_muo[i])**(1/3)
        kf_e  = (3*(np.pi**2)*n_ele[i])**(1/3)
        #constants as the density varies for baryons
        g_sig_NN = g_sat_NN[0] * f(nb, 0)
        g_sig_L0 = g_sat_L0[0]# * f(nb, 0)
        g_sgs_L0 = g_sat_L0[3]# * f(nb, 0)
        g_sig_Sp = g_sat_Sp[0]# * f(nb, 0)
        g_sig_S0 = g_sat_S0[0]# * f(nb, 0)
        g_sig_Sm = g_sat_Sm[0]# * f(nb, 0)
        g_sig_X0 = g_sat_X0[0]# * f(nb, 0)
        g_sig_Xm = g_sat_Xm[0]# * f(nb, 0)
        #effective masses
        m_eff_p  = m_n[0] - sigma[i]*g_sig_NN
        m_eff_n  = m_n[1] - sigma[i]*g_sig_NN
        m_eff_L0 = m_n[2] - sigma[i]*g_sig_L0 - sig_s[i]*g_sgs_L0
        m_eff_Sp = m_n[3] - sigma[i]*g_sig_Sp
        m_eff_S0 = m_n[4] - sigma[i]*g_sig_S0
        m_eff_Sm = m_n[5] - sigma[i]*g_sig_Sm
        m_eff_X0 = m_n[6] - sigma[i]*g_sig_X0
        m_eff_Xm = m_n[7] - sigma[i]*g_sig_Xm
        #constants derivatives as the density varies
        dg_sig_NN = g_sat_NN[0] * f(nb, 0, derive=True)
        dg_omg_NN = g_sat_NN[1] * f(nb, 1, derive=True)
        dg_rho_NN = g_sat_NN[2] * g(nb, derive=True)
        #nucleons pressure terms
        pre_p  = Inte(pre, kf_p, m_eff_p)
        pre_n  = Inte(pre, kf_n, m_eff_n)
        pre_L0 = Inte(pre, kf_L0, m_eff_L0)
        pre_Sp = Inte(pre, kf_Sp, m_eff_Sp)
        pre_S0 = Inte(pre, kf_S0, m_eff_S0)
        pre_Sm = Inte(pre, kf_Sm, m_eff_Sm)
        pre_X0 = Inte(pre, kf_X0, m_eff_X0)
        pre_Xm = Inte(pre, kf_Xm, m_eff_Xm)
        pre_BT = pre_p + pre_n + pre_L0 + pre_Sp + pre_S0 + pre_Sm + pre_X0 + pre_Xm
        #leptons pressure term
        pre_e = Inte(pre, kf_e, m_l[0])
        pre_m = Inte(pre, kf_m, m_l[1])
        #mesons derivative terms
        I_p = Inte(ns, kf_p, m_eff_p)
        I_n = Inte(ns, kf_n, m_eff_n)
        pre_m1  = dg_omg_NN*omega[i]*(n_neu[i]+n_pro[i])
        pre_m1 += dg_rho_NN*rho[i]  *(n_neu[i]-n_pro[i])
        pre_m1 -= dg_sig_NN*sigma[i]*(I_p+I_n)
        #mesons pressure term
        pre_m2  = -0.5*(m_m[0]**2*sigma[i]**2 - m_m[1]**2*omega[i]**2 - m_m[2]**2*rho[i]**2)
        pre_m2 += -0.5*(m_m[3]**2*sig_s[i]**2 - m_m[4]**2*phi[i]**2)
        #system's total pressure
        press[i] = (1/3)*(pre_BT + pre_e + pre_m) + nb*pre_m1 + pre_m2

    return press


#computation parameters
number_dens = 600
min_dens = 0.01
max_dens = 1.0

nb_dens = np.linspace(min_dens, max_dens, number_dens)

#array for the solutions
sigma = np.zeros(number_dens + 1)
omega = np.zeros(number_dens + 1)
rho   = np.zeros(number_dens + 1)
sig_s = np.zeros(number_dens + 1)
phi   = np.zeros(number_dens + 1)
n_pro = np.zeros(number_dens + 1)
n_neu = np.zeros(number_dens + 1)
n_ele = np.zeros(number_dens + 1)
n_muo = np.zeros(number_dens + 1)
n_lam = np.zeros(number_dens + 1)
n_Sgp = np.zeros(number_dens + 1)
n_Sg0 = np.zeros(number_dens + 1)
n_Sgm = np.zeros(number_dens + 1)
n_Xi0 = np.zeros(number_dens + 1)
n_Xim = np.zeros(number_dens + 1)

#initial values
sigma[0], omega[0], rho[0] = min_dens * g_sat_NN/m_m[0:3]**2
sig_s[0] = 0.0
phi[0]   = 0.0
n_pro[0] = 0.01 * min_dens
n_neu[0] = 0.99 * min_dens
n_ele[0] = 0.01 * min_dens
n_muo[0] = 0.00 * min_dens
n_lam[0] = 0.00 * min_dens
n_Sgp[0] = 0.00 * min_dens
n_Sg0[0] = 0.00 * min_dens
n_Sgm[0] = 0.00 * min_dens
n_Xi0[0] = 0.00 * min_dens
n_Xim[0] = 0.00 * min_dens


t0 = time.time()

muon    = False  #boolean variable becomes True if there are muons
Lambda0 = False  #boolean variable becomes True if there are lambdas
Sigma_p = False  #boolean variable becomes True if there are Sigma plus
Sigma_0 = False  #boolean variable becomes True if there are Sigma 0
Sigma_m = False  #boolean variable becomes True if there are Sigma minus
Xi_0    = False  #boolean variable becomes True if there are Xi 0
Xi_m    = False  #boolean variable becomes True if there are Xi minus

for i, nb in enumerate(nb_dens): #loop over densities
    #constants as the density varies for nucleon
    g_sig_NN = g_sat_NN[0] * f(nb, 0)
    g_omg_NN = g_sat_NN[1] * f(nb, 1)
    g_rho_NN = g_sat_NN[2] * g(nb)
    #constants as the density varies for Lambda
    g_sig_L0 = g_sat_L0[0]# * f(nb, 0)
    g_omg_L0 = g_sat_L0[1]# * f(nb, 1)
    g_phi_L0 = g_sat_L0[2]# * f(nb, 1)
    g_sgs_L0 = g_sat_L0[3]# * f(nb, 0)
    #constants as the density varies for Sigma plus
    g_sig_Sp = g_sat_Sp[0]# * f(nb, 0)
    g_omg_Sp = g_sat_Sp[1]# * f(nb, 1)
    g_rho_Sp = g_sat_Sp[2]# * g(nb)
    #constants as the density varies for Sigma 0
    g_sig_S0 = g_sat_S0[0]# * f(nb, 0)
    g_omg_S0 = g_sat_S0[1]# * f(nb, 1)
    #constants as the density varies for Sigma minus
    g_sig_Sm = g_sat_Sm[0]# * f(nb, 0)
    g_omg_Sm = g_sat_Sm[1]# * f(nb, 1)
    g_rho_Sm = g_sat_Sm[2]# * g(nb)
    #constants as the density varies for Xi 0
    g_sig_X0 = g_sat_X0[0]# * f(nb, 0)
    g_omg_X0 = g_sat_X0[1]# * f(nb, 1)
    g_rho_X0 = g_sat_X0[2]# * g(nb)
    #constants as the density varies for Xi minus
    g_sig_Xm = g_sat_Xm[0]# * f(nb, 0)
    g_omg_Xm = g_sat_Xm[1]# * f(nb, 1)
    g_rho_Xm = g_sat_Xm[2]# * g(nb)

    #starting point depending on i to speed up the computation
    start = (sigma[i], omega[i], rho[i],   sig_s[i], phi[i],
             n_neu[i], n_pro[i], n_ele[i], n_muo[i], n_lam[i],
             n_Sgp[i], n_Sg0[i], n_Sgm[i], n_Xi0[i], n_Xim[i])
    #we solve the system and keep the solution
    arg = (nb,
           g_sig_NN, g_omg_NN, g_rho_NN,
           g_sig_L0, g_omg_L0, g_phi_L0, g_sgs_L0,
           g_sig_Sp, g_omg_Sp, g_rho_Sp,
           g_sig_S0, g_omg_S0,
           g_sig_Sm, g_omg_Sm, g_rho_Sm,
           g_sig_X0, g_omg_X0, g_rho_X0,
           g_sig_Xm, g_omg_Xm, g_rho_Xm,
           muon, Lambda0, Sigma_p, Sigma_0, Sigma_m, Xi_0, Xi_m)

    sol = root(system , start, args=arg, tol=1e-10, method='hybr')

    sigma[i+1], omega[i+1], rho[i+1],   sig_s[i+1], phi[i+1]   = sol.x[0:5]
    n_neu[i+1], n_pro[i+1], n_ele[i+1], n_muo[i+1], n_lam[i+1] = abs(sol.x[5:10])
    n_Sgp[i+1], n_Sg0[i+1], n_Sgm[i+1], n_Xi0[i+1], n_Xim[i+1] = abs(sol.x[10:15])


    #always usefull quantities for particle's threshold
    #electron
    kf_e  = (3*(np.pi**2)*n_ele[i+1])**(1/3)
    mu_e  = np.sqrt(kf_e**2 + m_l[0]**2)
    #neutron
    kf_n  = (3*(np.pi**2)*n_neu[i+1])**(1/3)
    mu_n  = np.sqrt(kf_n**2 + (m_n[1] - sigma[i+1]*g_sig_NN)**2)
    mu_n += g_rho_NN*rho[i+1] + g_omg_NN*omega[i+1]
    #proton
    kf_p  = (3*(np.pi**2)*n_pro[i+1])**(1/3)
    mu_p  = np.sqrt(kf_p**2 + (m_n[0] - sigma[i+1]*g_sig_NN)**2)
    mu_p += - g_rho_NN*rho[i+1] + g_omg_NN*omega[i+1]


    #comtutation of particle's threshold
    #not all particles necessarily appear
    #to "turn off" a particle it is enough
    #not to calculate its threshold
    #commenting the appropiate line

    #Calculation of the Fermi moment squared for muons
    kf_m_2 = mu_e**2 - m_l[1]**2
    #if negative they are below the threshold and there are no muons
    if kf_m_2 <= 0:
        muon = False
        n_muo[i+1] = 0
    else:
        muon = True


    #Calculation of the Fermi moment squared for lambdas
    mu_L_2 = (mu_n - g_omg_L0*omega[i+1] - g_phi_L0*phi[i+1])**2
    kf_L_2 = mu_L_2 - (m_n[2] - sigma[i+1]*g_sig_L0 - sig_s[i+1]*g_sgs_L0)**2
    #if negative they are below the threshold and there are no lambda
    if kf_L_2 <= 0:
        Lambda0 = False
        n_lam[i+1] = 0
    else:
        Lambda0 = True


    #Calculation of the Fermi moment squared for Sigma_p
    mu_Sp_2 = (mu_p - g_omg_Sp*omega[i+1] + g_rho_Sp*rho[i+1])**2
    kf_Sp_2 = mu_Sp_2 - (m_n[3] - sigma[i+1]*g_sig_Sp)**2
    #if negative they are below the threshold and there are no Sigma_p
    if kf_Sp_2 <= 0:
        Sigma_p = False
        n_Sgp[i+1] = 0
    else:
        Sigma_p = True


    #Calculation of the Fermi moment squared for Sigma_0
    mu_S0_2 = (mu_n - g_omg_S0*omega[i+1])**2
    kf_S0_2 = mu_S0_2 - (m_n[4] - sigma[i+1]*g_sig_S0)**2
    #if negative they are below the threshold and there are no Sigma_0
    if kf_S0_2 <= 0:
        Sigma_0 = False
        n_Sg0[i+1] = 0
    else:
        Sigma_0 = True


    #Calculation of the Fermi moment squared for Sigma_m
    mu_Sm_2 = (mu_n + mu_e - g_omg_Sm*omega[i+1] - g_rho_Sm*rho[i+1])**2
    kf_Sm_2 = mu_Sm_2 - (m_n[5] - sigma[i+1]*g_sig_Sm)**2
    #if negative they are below the threshold and there are no Sigma_m
    if kf_Sm_2 <= 0:
        Sigma_m = False
        n_Sgm[i+1] = 0
    else:
        Sigma_m = True


    #Calculation of the Fermi moment squared for Xi_0
    mu_X0_2 = (mu_n - g_omg_X0*omega[i+1] + g_rho_X0*rho[i+1])**2
    kf_X0_2 = mu_X0_2 - (m_n[6] - sigma[i+1]*g_sig_X0)**2
    #if negative they are below the threshold and there are no Xi_0
    if kf_X0_2 <= 0:
        Xi_0 = False
        n_Xi0[i+1] = 0
    else:
        Xi_0    = True


    #Calculation of the Fermi moment squared for Xi_m
    mu_Xm_2 = (mu_n + mu_e - g_omg_Xm*omega[i+1] - g_rho_Xm*rho[i+1])**2
    kf_Xm_2 = mu_Xm_2 - (m_n[7] - sigma[i+1]*g_sig_Xm)**2
    #if negative they are below the threshold and there are no Xi_m
    if kf_Xm_2 <= 0:
        Xi_m = False
        n_Xim[i+1] = 0
    else:
        Xi_m = True



#we delete the initial value
sigma = sigma[1:]
omega = omega[1:]
rho   = rho[1:]
sig_s = sig_s[1:]
phi   = phi[1:]
n_pro = n_pro[1:]
n_neu = n_neu[1:]
n_ele = n_ele[1:]
n_muo = n_muo[1:]
n_lam = n_lam[1:]
n_Sgp = n_Sgp[1:]
n_Sg0 = n_Sg0[1:]
n_Sgm = n_Sgm[1:]
n_Xi0 = n_Xi0[1:]
n_Xim = n_Xim[1:]

#energy and pressure computation
ene_nb =total_energy(nb_dens,
                     n_pro, n_neu, n_lam, n_Sgp, n_Sg0, n_Sgm, n_Xi0, n_Xim,
                     n_ele, n_muo, sigma, omega, rho, sig_s, phi)

pre_nb = total_pressure(nb_dens,
                        n_pro, n_neu, n_lam, n_Sgp, n_Sg0, n_Sgm, n_Xi0, n_Xim,
                        n_ele, n_muo, sigma, omega, rho, sig_s, phi)

#enrgy per nucleon
E_over_A =  hbc*(ene_nb/nb_dens)
#pressure
pre_nb *= hbc

dt = time.time() - t0
print(f'Elapsed time: {dt}')

#abundances plot
plt.figure(1)
plt.title('Abundance of each species')
plt.xlabel('density [$fm^{-3}$]')
plt.plot(nb_dens, n_pro/nb_dens, 'red',   )
plt.plot(nb_dens, n_neu/nb_dens, 'black'  )
plt.plot(nb_dens, n_ele/nb_dens, 'blue'   )
plt.plot(nb_dens, n_muo/nb_dens, 'purple' )
plt.plot(nb_dens, n_lam/nb_dens, 'green'  )
plt.plot(nb_dens, n_Sgp/nb_dens, 'orange' )
plt.plot(nb_dens, n_Sg0/nb_dens, 'lime'   )
plt.plot(nb_dens, n_Sgm/nb_dens, 'pink'   )
plt.plot(nb_dens, n_Xi0/nb_dens, 'navy'   )
plt.plot(nb_dens, n_Xim/nb_dens, 'fuchsia')

plt.annotate('n',          (0.30, 1.00877))
plt.annotate('p',          (0.30, 0.10122))
plt.annotate('e',          (0.30, 0.06802))
plt.annotate('$\mu$',      (0.30, 0.03320))
plt.annotate('$\Lambda$',  (0.43, 0.13768))
plt.annotate('$\Sigma^-$', (0.40, 0.03405))
plt.annotate('$\Xi^0$',    (0.75, 0.07159), fontsize=12)
plt.annotate('$\Xi^-$',    (0.63, 0.10564), fontsize=12)
plt.ylim(1e-3,1.3)
plt.yscale('log')
plt.grid()
#plt.savefig('abbondanze')

#energy plot
plt.figure(2)
plt.title('Energy')
plt.ylabel('E [Mev fm$^{-3}$]')
plt.xlabel('density [$fm^{-3}$]')
plt.grid()
plt.plot(nb_dens, E_over_A, 'k')
#plt.savefig('energia')

#pressure plot
plt.figure(3)
plt.title('Pressure')
plt.ylabel('P [Mev/$fm^3$]')
plt.xlabel('density [$fm^{-3}$]')
plt.grid()
plt.plot(nb_dens, pre_nb, 'k')
#plt.savefig('pressione')

plt.show()
