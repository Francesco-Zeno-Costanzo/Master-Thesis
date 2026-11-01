"""
This code solves the equations of the relativistic mean field model.
The calculation are done at constant entropy per baryon S/A = const.
The particles considered are all baryons of baryon octet, muons, electrons and neutrinos.
If you want you can consider all coupling constants density-dependent or not.
"""
import time
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from scipy.optimize import root

#==================================================================
# Computation parameters
#==================================================================

number_dens = 84
min_dens    = 0.1
max_dens    = 1

nb_dens = np.logspace(np.log10(min_dens), np.log10(max_dens), number_dens)

#nb_dens = np.linspace(min_dens, max_dens, number_dens)

#entropy per baryon in unit of K_B
S = 2

#==================================================================
# density-dependet coupling-constant
#==================================================================

const_hyp = False  #coupling constant dependent from density or not
const_nuc = False  #coupling constant dependent from density or not

#==================================================================
# Path of file for saving data
#==================================================================

SAVE  = True # If False the data are not saved
path1 = fr"abb_HYPn_S{S}_dd.txt"
path2 = fr"pot_HYPn_S{S}_dd.txt"
path3 = fr"EOS_HYPn_S{S}_dd.txt"

#==================================================================
# Physical constant and coupling constant
#==================================================================

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
m_m   = np.array([m_sig, m_omg, m_rho, m_sgs, m_phi])/hbc

#lepton masses
m_el = 0.51099907
m_mu = 105.6583755
m_nu = 0
m_l  = np.array([m_el, m_mu, m_nu])/hbc

#coupling constants at nb_sat
g_sg_sat = 10.5396
g_om_sat = 13.0189
g_rh_sat = 3.6836 #we already considered 1/2
g_sat_NN = np.array([g_sg_sat, g_om_sat, g_rh_sat])

#hyperonic couplig costant Lambda-meson
L_sig_DD = 0.65869756
L_sig_ND = 0.61498048
L_sgs_DD = 0.535
L_sgs_ND = 0.538
x_sig_L0 = L_sig_ND
x_omg_L0 = 2/3
x_phi_L0 = - np.sqrt(2)/3
x_sgs_L0 = L_sgs_ND

g_sig_L_sat = x_sig_L0 * g_sg_sat
g_omg_L_sat = x_omg_L0 * g_om_sat
g_phi_L_sat = x_phi_L0 * g_om_sat
g_sgs_L_sat = x_sgs_L0 * g_sg_sat
#in order            sigma,       omega,       phi           sigma*
g_sat_L0 = np.array([g_sig_L_sat, g_omg_L_sat, g_phi_L_sat, g_sgs_L_sat])

#hyperonic couplig costant Sigma_p-meson
S_sig_30_DD = 0.50447765
S_sig_30_ND = 0.46076056
S_sig_00_DD = 0.57910019
S_sig_00_ND = 0.5353831
x_sig_S = S_sig_00_ND
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
X_sig_DD = 0.37988278
X_sig_ND = 0.30251540
x_sig_X = X_sig_ND
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

#==================================================================
# Initial Value
#==================================================================

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
n_nuu = np.zeros(number_dens + 1)
mu_pr = np.zeros(number_dens + 1)
mu_ne = np.zeros(number_dens + 1)
mu_el = np.zeros(number_dens + 1)
mu_mu = np.zeros(number_dens + 1)
mu_L0 = np.zeros(number_dens + 1)
mu_Sp = np.zeros(number_dens + 1)
mu_S0 = np.zeros(number_dens + 1)
mu_Sm = np.zeros(number_dens + 1)
mu_X0 = np.zeros(number_dens + 1)
mu_Xm = np.zeros(number_dens + 1)
mu_nu = np.zeros(number_dens + 1)
T_nbd = np.zeros(number_dens + 1)


#initial values
sigma[0], omega[0], rho[0] = g_sat_NN/m_m[0:3]**2 * min_dens
sig_s[0], phi[0] = g_sat_L0[2:4]/m_m[3:5]**2 * min_dens
mu_pr[0], mu_ne[0], mu_L0[0], mu_Sp[0] = m_n[0:4]
mu_S0[0], mu_Sm[0], mu_X0[0], mu_Xm[0] = m_n[4:8]
mu_el[0] = m_l[0]
mu_mu[0] = m_l[1]
mu_nu[0] = m_l[2]
n_pro[0] = 0.1 * min_dens
n_neu[0] = 0.9 * min_dens
n_ele[0] = 0.1 * min_dens
n_muo[0] = 0.0 #* min_dens
n_nuu[0] = 0.0 #* min_dens
n_lam[0] = 0.0 #* min_dens
n_Sgp[0] = 0.0 #* min_dens
n_Sg0[0] = 0.0 #* min_dens
n_Sgm[0] = 0.0 #* min_dens
n_Xi0[0] = 0.0 #* min_dens
n_Xim[0] = 0.0 #* min_dens
T_nbd[0] = 11 / hbc

#==================================================================
# Auxiliary Function
#==================================================================

def f(n, i, const, derive=False):
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
        i = 0 sigma or sigma*
        i = 1 omega or phi
    const : boolean
        if true we return 1 for function and zero for derivative
    derive : boolean
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

    if const:
        if not derive :
            return np.ones(len(n))
        else :
            return np.zeros(len(n))
    else :
        if not derive :
            num = 1 + b[i]*(x + d[i])**2
            den = 1 + c[i]*(x + d[i])**2
            g_n = a[i] * num/den
            return g_n

        else :
            num = 2*a[i]*(b[i] - c[i])*nb_sat**2 * (n + d[i]*nb_sat)
            den = (nb_sat**2 + c[i]*(n + d[i]*nb_sat)**2)**2
            dg_n = num/den
            return dg_n


def g(n, const, derive=False):
    """
    function that models the behaviour of the
    coupling constant depending on nb
    for the rho meson

    Parameters
    ----------
    x : float or 1darray
        nb/nb_sat
    const : boolean
        if true we return 1 for function and zero for derivative
    derive : boolean
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

    if const:
        if not derive :
            return np.ones(len(n))
        else :
            return np.zeros(len(n))
    else:
        if not derive :
            g_n = np.exp(-a[2]*(x-1))
            return g_n

        else :
            dg_n = - a[2]/nb_sat * np.exp(a[2]*(1-x))
            return dg_n


def fermi_dirac_p(k, M, Ene, mu, T, mis):
    """
    Fermi–Dirac statistics for particle

    Parameters
    ----------
    k : float
        momentum, integration variable
    M : float
        mass of particle
    Ene : float
        term of energy of meson in fermi-dirac
    mu : float
        chemical potential
    T : float
        temperature
    mis : boolean
         Fermi–Dirac statistics multiplies all integrals
         so the integration misure is included only if
         we want integrate Fermi–Dirac statistics alone
         mis = True -> \int k^2 dk/pi^2  1/(exp(...) + 1)
         mis = False-> \int k^2 dk/pi^2 f(...) 1/(exp(...) + 1)

    Return
    ------
    f_d : float
        Fermi–Dirac statistics for particle
    """

    if mis == True:

        ene_s = np.sqrt(k**2 + M**2) #single particles energy
        Ene_t = Ene + ene_s          #total energy
        f_d = 1/np.pi**2 * k**2 * np.exp(-(Ene_t - mu)/T)/(1 + np.exp(-(Ene_t - mu)/T))

        return f_d

    if mis == False:

        ene_s = np.sqrt(k**2 + M**2) #single particles energy
        Ene_t = Ene + ene_s          #total energy
        f_d = np.exp(-(Ene_t - mu)/T)/(1 + np.exp(-(Ene_t - mu)/T))

        return f_d


def fermi_dirac_a(k, M, Ene, mu, T, mis):
    """
    Fermi–Dirac statistics for anti-particle

    Parameters
    ----------
    k : float
        momentum, integration variable
    M : float
        mass of particle
    Ene : float
        term of energy of meson in fermi-dirac
    mu : float
        chemical potential
    T : float
        temperature
    mis : boolean
         Fermi–Dirac statistics multiplies all integrals
         so the integration misure is included only if
         we want integrate Fermi–Dirac statistics alone
         mis = True -> \int k^2 dk/pi^2  1/(exp(...) + 1)
         mis = False-> \int k^2 dk/pi^2 f(...) 1/(exp(...) + 1)

    Return
    ------
    f_d : float
        Fermi–Dirac statistics for anti-particle
    """

    if mis == True:

        ene_s = np.sqrt(k**2 + M**2) #single particles energy
        Ene_t = Ene - ene_s          #total energy
        f_d = 1/np.pi**2 * k**2 * np.exp((Ene_t - mu)/T)/(1 + np.exp((Ene_t - mu)/T))

        return f_d

    if mis == False:

        ene_s = np.sqrt(k**2 + M**2)#single particles energy
        Ene_t = Ene - ene_s          #total energy
        f_d = np.exp((Ene_t - mu)/T)/(1 + np.exp((Ene_t - mu)/T))

        return f_d


def ns(k, M, Ene, mu, T):
    """
    integral for the sigma equation

    Parameters
    ----------
    k : float
        momentum, integration variable
    M : float
        effective mass m - g_sg * sigma
    Ene : float
        term of energy of meson in fermi-dirac
    mu : float
        chemical potential
    T : float
        temperature

    Return
    ----------
    x_dot : float
        integrand function
    """

    num = M * (fermi_dirac_p(k, M, Ene, mu, T, False) + fermi_dirac_a(k, M, Ene, mu, T, False))
    den = np.sqrt(k**2 + M**2)

    x_dot = 1/np.pi**2 * k**2 * num/den

    return x_dot


def ene(k, M, Ene, mu, T):
    """
    integral for the nucleons energy

    Parameters
    ----------
    k : float
        momentum, integration variable
    M : float
        mass of particle
    Ene : float
        term of energy of meson in fermi-dirac
    mu : float
        chemical potential
    T : float
        temperature

    Return
    ------
    x_dot : float
        integrand function
    """

    eps = 1/np.pi**2 * k**2 * np.sqrt(k**2 + M**2)
    x_dot = eps * (fermi_dirac_p(k, M, Ene, mu, T, False) + fermi_dirac_a(k, M, Ene, mu, T, False))

    return x_dot


def pre(k, M, Ene, mu, T):
    """
    integral for the nucleons pressure

    Parameters
    ----------
    k : float
        momentum, integration variable
    M : float
        mass of particle
    Ene : float
        term of energy of meson in fermi-dirac
    mu : float
        chemical potential
    T : float
        temperature

    Return
    ------
    x_dot : float
        integrand function
    """
    eps = 1/np.pi**2 * k**4 * 1 / np.sqrt(k**2 + M**2)
    x_dot = eps * (fermi_dirac_p(k, M, Ene, mu, T, False) + fermi_dirac_a(k, M, Ene, mu, T, False))

    return x_dot


def entropy(k, M, Ene, mu, T):
    """
    integral for the entropy

    Parameters
    ----------
    k : float
        momentum, integration variable
    M : float
        mass of particle
    Ene : float
        term of energy of meson in fermi-dirac
    mu : float
        chemical potential
    T : float
        temperature

    Return
    ------
    x_dot : float
        integrand function
    """
    
    # Particle entropy
    entr_p_1 = fermi_dirac_p(k, M, Ene, mu, T, False)

    if entr_p_1 < 1e-16:
        entr_p_1 = 0
        entr_p_2 = 0
    else :
        entr_p_2 = np.log(entr_p_1)

    entr_p_3 = 1 - fermi_dirac_p(k, M, Ene, mu, T, False)

    if entr_p_3 < 1e-16:
        entr_p_3 = 0
        entr_p_4 = 0
    else :
        entr_p_4 = np.log(entr_p_3)

    entropia_p = entr_p_1 * entr_p_2 + entr_p_3 * entr_p_4
    
    # Anti particle entropy
    entr_a_1 = fermi_dirac_a(k, M, Ene, mu, T, False)

    if entr_a_1 < 1e-16:
        entr_a_1 = 0
        entr_a_2 = 0
    else :
        entr_a_2 = np.log(entr_a_1)

    entr_a_3 = 1 - fermi_dirac_a(k, M, Ene, mu, T, False)

    if entr_a_3 < 1e-16:
        entr_a_3 = 0
        entr_a_4 = 0
    else :
        entr_a_4 = np.log(entr_a_3)

    entropia_a = entr_a_1 * entr_a_2 + entr_a_3 * entr_a_4

    x_dot = 1/np.pi**2 * k**2 * (entropia_p + entropia_a)

    return x_dot


def Inte(f, arg):
    """
    this function computes the integral via scipy

    Parameters
    ----------
    f : function
        integrand function
    arg : tule
        extra arguments to pass to f

    Return
    -----------
    I : float
        \int_0^cur_off f(k, arg) dk
    """

    I, dI = integrate.quad(f, 0, 19.0, args=arg)

    return I


#==============================================================
# System to solve: all octet + elecntrons and muons
#==============================================================


def system(V, nb,
           g_sig_NN,  g_omg_NN,  g_rho_NN,
           dg_sig_NN, dg_omg_NN, dg_rho_NN,
           S,
           g_sig_L0,  g_omg_L0,  g_phi_L0,  g_sgs_L0,
           dg_sig_L0, dg_omg_L0, dg_phi_L0, dg_sgs_L0,
           g_sig_Sp,  g_omg_Sp,  g_rho_Sp,
           dg_sig_Sp, dg_omg_Sp, dg_rho_Sp,
           g_sig_S0,  g_omg_S0,
           dg_sig_S0, dg_omg_S0,
           g_sig_Sm,  g_omg_Sm,  g_rho_Sm,
           dg_sig_Sm, dg_omg_Sm, dg_rho_Sm,
           g_sig_X0,  g_omg_X0,  g_rho_X0,
           dg_sig_X0, dg_omg_X0, dg_rho_X0,
           g_sig_Xm,  g_omg_Xm,  g_rho_Xm,
           dg_sig_Xm, dg_omg_Xm, dg_rho_Xm):
    '''
    system to solve

    Parameters
    ----------
    V : 1darray
        array of the variables
    nb : float
        total baryonic density
    S : float
        Entropy
    g_meson_baryon : float
        coupling constant
    dg_meson_baryon : float
        derivative of coupling constant

    Returns
    -------
    list of equations to solve
    ri = 0
    '''
    sig, omg, rho                   = V[0:3]
    p, n, e, mu_p, mu_n, mu_e       = V[3:9]
    m, mu_m, sgs, phi, L0, mu_L0    = V[9:15]
    Sp, S0, Sm, mu_Sp, mu_S0, mu_Sm = V[15:21]
    X0, Xm, mu_X0, mu_Xm, T         = V[21:26]
    nu, mu_nu                       = V[26:28]

    #effective masses
    m_eff_p  = m_n[0] - sig*g_sig_NN
    m_eff_n  = m_n[1] - sig*g_sig_NN
    m_eff_L0 = m_n[2] - sig*g_sig_L0 - sgs*g_sgs_L0
    m_eff_Sp = m_n[3] - sig*g_sig_Sp
    m_eff_S0 = m_n[4] - sig*g_sig_S0
    m_eff_Sm = m_n[5] - sig*g_sig_Sm
    m_eff_X0 = m_n[6] - sig*g_sig_X0
    m_eff_Xm = m_n[7] - sig*g_sig_Xm
    #energy terms
    Sigma_Np = g_omg_NN*omg - g_rho_NN*rho
    Sigma_Nn = g_omg_NN*omg + g_rho_NN*rho
    Sigma_L0 = g_omg_L0*omg + g_phi_L0*phi
    Sigma_Sp = g_omg_Sp*omg - g_rho_Sp*rho
    Sigma_S0 = g_omg_S0*omg
    Sigma_Sm = g_omg_Sm*omg + g_rho_Sm*rho
    Sigma_X0 = g_omg_X0*omg - g_rho_X0*rho
    Sigma_Xm = g_omg_Xm*omg + g_rho_Xm*rho
    #rearrangemet terms
    Sigma_r_NN  = (dg_omg_NN/g_omg_NN)*(m_m[1]*omg)**2
    Sigma_r_NN += (dg_rho_NN/g_rho_NN)*(m_m[2]*rho)**2
    Sigma_r_NN -= (dg_sig_NN/g_sig_NN)*(m_m[0]*sig)**2
    #
    Sigma_r_L0  = (dg_omg_L0/g_omg_L0)*(m_m[1]*omg)**2
    Sigma_r_L0 -= (dg_sig_L0/g_sig_L0)*(m_m[0]*sig)**2
    Sigma_r_L0 -= (dg_sgs_L0/g_sgs_L0)*(m_m[3]*sgs)**2
    Sigma_r_L0 += (dg_phi_L0/g_phi_L0)*(m_m[4]*phi)**2
    #
    Sigma_r_Sp  = (dg_omg_Sp/g_omg_Sp)*(m_m[1]*omg)**2
    Sigma_r_Sp += (dg_rho_Sp/g_rho_Sp)*(m_m[2]*rho)**2
    Sigma_r_Sp -= (dg_sig_Sp/g_sig_Sp)*(m_m[0]*sig)**2
    #
    Sigma_r_S0  = (dg_omg_S0/g_omg_S0)*(m_m[1]*omg)**2
    Sigma_r_S0 -= (dg_sig_S0/g_sig_S0)*(m_m[0]*sig)**2
    #
    Sigma_r_Sm  = (dg_omg_Sm/g_omg_Sm)*(m_m[1]*omg)**2
    Sigma_r_Sm += (dg_rho_Sm/g_rho_Sm)*(m_m[2]*rho)**2
    Sigma_r_Sm -= (dg_sig_Sm/g_sig_Sm)*(m_m[0]*sig)**2
    #
    Sigma_r_X0  = (dg_omg_X0/g_omg_X0)*(m_m[1]*omg)**2
    Sigma_r_X0 += (dg_rho_X0/g_rho_X0)*(m_m[2]*rho)**2
    Sigma_r_X0 -= (dg_sig_X0/g_sig_X0)*(m_m[0]*sig)**2
    #
    Sigma_r_Xm  = (dg_omg_Xm/g_omg_Xm)*(m_m[1]*omg)**2
    Sigma_r_Xm += (dg_rho_Xm/g_rho_Xm)*(m_m[2]*rho)**2
    Sigma_r_Xm -= (dg_sig_Xm/g_sig_Xm)*(m_m[0]*sig)**2
    #numerical integrals for sigma
    I_p  = Inte(ns, (m_eff_p , Sigma_Np + Sigma_r_NN, mu_p , T))
    I_n  = Inte(ns, (m_eff_n , Sigma_Nn + Sigma_r_NN, mu_n , T))
    I_L0 = Inte(ns, (m_eff_L0, Sigma_L0 + Sigma_r_L0, mu_L0, T))
    I_Sp = Inte(ns, (m_eff_Sp, Sigma_Sp + Sigma_r_Sp, mu_Sp, T))
    I_S0 = Inte(ns, (m_eff_S0, Sigma_S0 + Sigma_r_S0, mu_S0, T))
    I_Sm = Inte(ns, (m_eff_Sm, Sigma_Sm + Sigma_r_Sm, mu_Sm, T))
    I_X0 = Inte(ns, (m_eff_X0, Sigma_X0 + Sigma_r_X0, mu_X0, T))
    I_Xm = Inte(ns, (m_eff_Xm, Sigma_Xm + Sigma_r_Xm, mu_Xm, T))
    #total integrals for sigma
    I_sg_n  = g_sig_NN*I_p  + g_sig_NN*I_n  + g_sig_L0*I_L0 + g_sig_Sp*I_Sp
    I_sg_n += g_sig_S0*I_S0 + g_sig_Sm*I_Sm + g_sig_X0*I_X0 + g_sig_Xm*I_Xm
    #density for omega
    ND_omg  = g_omg_NN*n  + g_omg_NN*p  + g_omg_L0*L0 + g_omg_Sp*Sp
    ND_omg += g_omg_S0*S0 + g_omg_Sm*Sm + g_omg_X0*X0 + g_omg_Xm*Xm
    #density for rho
    ND_rho = g_rho_NN*(n - p) - g_rho_Sp*Sp + g_rho_Sm*Sm - g_rho_X0*X0 + g_rho_Xm*Xm
    #total integrals for sigma*
    I_sg_s = g_sgs_L0*I_L0
    #density for phi
    ND_phi = g_phi_L0*L0
    #fermi dirac integrals baryons
    I_f_p_p  = Inte(fermi_dirac_p, (m_eff_p , Sigma_Np + Sigma_r_NN, mu_p , T, True))
    I_f_p_a  = Inte(fermi_dirac_a, (m_eff_p , Sigma_Np + Sigma_r_NN, mu_p , T, True))
    I_f_n_p  = Inte(fermi_dirac_p, (m_eff_n , Sigma_Nn + Sigma_r_NN, mu_n , T, True))
    I_f_n_a  = Inte(fermi_dirac_a, (m_eff_n , Sigma_Nn + Sigma_r_NN, mu_n , T, True))
    I_f_L0_p = Inte(fermi_dirac_p, (m_eff_L0, Sigma_L0 + Sigma_r_L0, mu_L0, T, True))
    I_f_L0_a = Inte(fermi_dirac_a, (m_eff_L0, Sigma_L0 + Sigma_r_L0, mu_L0, T, True))
    I_f_Sp_p = Inte(fermi_dirac_p, (m_eff_Sp, Sigma_Sp + Sigma_r_Sp, mu_Sp, T, True))
    I_f_Sp_a = Inte(fermi_dirac_a, (m_eff_Sp, Sigma_Sp + Sigma_r_Sp, mu_Sp, T, True))
    I_f_S0_p = Inte(fermi_dirac_p, (m_eff_S0, Sigma_S0 + Sigma_r_S0, mu_S0, T, True))
    I_f_S0_a = Inte(fermi_dirac_a, (m_eff_S0, Sigma_S0 + Sigma_r_S0, mu_S0, T, True))
    I_f_Sm_p = Inte(fermi_dirac_p, (m_eff_Sm, Sigma_Sm + Sigma_r_Sm, mu_Sm, T, True))
    I_f_Sm_a = Inte(fermi_dirac_a, (m_eff_Sm, Sigma_Sm + Sigma_r_Sm, mu_Sm, T, True))
    I_f_X0_p = Inte(fermi_dirac_p, (m_eff_X0, Sigma_X0 + Sigma_r_X0, mu_X0, T, True))
    I_f_X0_a = Inte(fermi_dirac_a, (m_eff_X0, Sigma_X0 + Sigma_r_X0, mu_X0, T, True))
    I_f_Xm_p = Inte(fermi_dirac_p, (m_eff_Xm, Sigma_Xm + Sigma_r_Xm, mu_Xm, T, True))
    I_f_Xm_a = Inte(fermi_dirac_a, (m_eff_Xm, Sigma_Xm + Sigma_r_Xm, mu_Xm, T, True))
    #fermi dirac integrals for leptons
    I_f_e_p  = Inte(fermi_dirac_p, (m_l[0], 0, mu_e,  T, True))
    I_f_e_a  = Inte(fermi_dirac_a, (m_l[0], 0, mu_e,  T, True))
    I_f_m_p  = Inte(fermi_dirac_p, (m_l[1], 0, mu_m,  T, True))
    I_f_m_a  = Inte(fermi_dirac_a, (m_l[1], 0, mu_m,  T, True))
    I_f_nu_p = Inte(fermi_dirac_p, (m_l[2], 0, mu_nu, T, True))
    I_f_nu_a = Inte(fermi_dirac_a, (m_l[2], 0, mu_nu, T, True))
    #baryons entropy
    I_e_p  = Inte(entropy, (m_eff_p,  Sigma_Np + Sigma_r_NN, mu_p,  T))
    I_e_n  = Inte(entropy, (m_eff_n,  Sigma_Nn + Sigma_r_NN, mu_n,  T))
    I_e_L0 = Inte(entropy, (m_eff_L0, Sigma_L0 + Sigma_r_L0, mu_L0, T))
    I_e_Sp = Inte(entropy, (m_eff_Sp, Sigma_Sp + Sigma_r_Sp, mu_Sp, T))
    I_e_S0 = Inte(entropy, (m_eff_S0, Sigma_S0 + Sigma_r_S0, mu_S0, T))
    I_e_Sm = Inte(entropy, (m_eff_Sm, Sigma_Sm + Sigma_r_Sm, mu_Sm, T))
    I_e_X0 = Inte(entropy, (m_eff_X0, Sigma_X0 + Sigma_r_X0, mu_X0, T))
    I_e_Xm = Inte(entropy, (m_eff_Xm, Sigma_Xm + Sigma_r_Xm, mu_Xm, T))
    I_e_B = I_e_p + I_e_n + I_e_L0 + I_e_Sp + I_e_S0 + I_e_Sm + I_e_X0 + I_e_Xm
    #lepton term
    I_e_e  = Inte(entropy, (m_l[0], 0, mu_e,  T))
    I_e_m  = Inte(entropy, (m_l[1], 0, mu_m,  T))
    I_e_nu = Inte(entropy, (m_l[2], 0, mu_nu, T))
    #equation to solve
    r1 = sig - 1/m_m[0]**2 * I_sg_n
    r2 = omg - 1/m_m[1]**2 * ND_omg
    r3 = rho - 1/m_m[2]**2 * ND_rho
    r4 = sgs - 1/m_m[3]**2 * I_sg_s
    r5 = phi - 1/m_m[4]**2 * ND_phi
    r6 = nb - (n + p + L0 + Sp + S0 + Sm + X0 + Xm)
    r7 = (p + Sp) - (e + m + Sm + Xm)
    r8  = mu_n - mu_p - mu_e + mu_nu
    r9  = mu_e - mu_m - mu_nu
    r10 = mu_n - mu_L0
    r11 = mu_p - mu_Sp
    r12 = mu_n - mu_S0
    r13 = mu_n + mu_e - mu_Sm - mu_nu
    r14 = mu_n - mu_X0
    r15 = mu_n + mu_e - mu_Xm - mu_nu
    r16 = p - (I_f_p_p - I_f_p_a)
    r17 = n - (I_f_n_p - I_f_n_a)
    r18 = e - (I_f_e_p - I_f_e_a)
    r19 = m - (I_f_m_p - I_f_m_a)
    r20 = L0 - (I_f_L0_p - I_f_L0_a)
    r21 = Sp - (I_f_Sp_p - I_f_Sp_a)
    r22 = S0 - (I_f_S0_p - I_f_S0_a)
    r23 = Sm - (I_f_Sm_p - I_f_Sm_a)
    r24 = X0 - (I_f_X0_p - I_f_X0_a)
    r25 = Xm - (I_f_Xm_p - I_f_Xm_a)
    r26 = S + (I_e_B + I_e_e + I_e_m + I_e_nu)/nb
    r27 = nu - (I_f_nu_p - I_f_nu_a)
    r28 = nu + e - 0.33*nb

    R = [r1,  r2,  r3,  r4,  r5,  r6,  r7,  r8,  r9,  r10,
         r11, r12, r13, r14, r15, r16, r17, r18, r19, r20,
         r21, r22, r23, r24, r25, r26, r27, r28
        ]

    return R

#==================================================================
# Function for termodinamical quantities
#==================================================================

def total_energy(nb_dens, sigma, omega, rho, phi, sig_s,
                 mu_pr, mu_ne, mu_L0, mu_Sp, mu_S0, mu_Sm,
                 mu_X0, mu_Xm, mu_el, mu_mu, mu_nu, T,
                 g_sig_NN,  g_omg_NN,  g_rho_NN,
                 g_sig_L0,  g_omg_L0,  g_phi_L0, g_sgs_L0,
                 g_sig_Sp,  g_omg_Sp,  g_rho_Sp,
                 g_sig_S0,  g_omg_S0,
                 g_sig_Sm,  g_omg_Sm,  g_rho_Sm,
                 g_sig_X0,  g_omg_X0,  g_rho_X0,
                 g_sig_Xm,  g_omg_Xm,  g_rho_Xm,
                 dg_sig_NN, dg_omg_NN, dg_rho_NN,
                 dg_sig_L0, dg_omg_L0, dg_phi_L0, dg_sgs_L0,
                 dg_sig_Sp, dg_omg_Sp, dg_rho_Sp,
                 dg_sig_S0, dg_omg_S0,
                 dg_sig_Sm, dg_omg_Sm, dg_rho_Sm,
                 dg_sig_X0, dg_omg_X0, dg_rho_X0,
                 dg_sig_Xm, dg_omg_Xm, dg_rho_Xm
                 ):
    '''
    Computation of the system's total energy

    Parameters
    ----------
    nb_dens : 1darray
        total baryonic density
    sigma, omega, rho, sig_s, phi : 1darray
        array for the mesons solutions
    mu_pr, ..., mu_mu : 1darrray
        array with chemical potential for each species
    T : 1darray
        temperature
    g_sig_NN, ..., dg_rho_Xm : 1darray
        array with the value of coupling constant

    Retunrs
    -------
    energ : 1darray
        array for the total energy
    '''
    number_dens = len(nb_dens)
    #array for the energy
    energ = np.zeros(number_dens)

    for i, nb in enumerate(nb_dens):

        #effective masses
        m_eff_p  = m_n[0] - sigma[i]*g_sig_NN[i]
        m_eff_n  = m_n[1] - sigma[i]*g_sig_NN[i]
        m_eff_L0 = m_n[2] - sigma[i]*g_sig_L0[i] - sig_s[i]*g_sgs_L0[i]
        m_eff_Sp = m_n[3] - sigma[i]*g_sig_Sp[i]
        m_eff_S0 = m_n[4] - sigma[i]*g_sig_S0[i]
        m_eff_Sm = m_n[5] - sigma[i]*g_sig_Sm[i]
        m_eff_X0 = m_n[6] - sigma[i]*g_sig_X0[i]
        m_eff_Xm = m_n[7] - sigma[i]*g_sig_Xm[i]
        #energy terms for baryons
        Sigma_Np = g_omg_NN[i]*omega[i] - g_rho_NN[i]*rho[i]
        Sigma_Nn = g_omg_NN[i]*omega[i] + g_rho_NN[i]*rho[i]
        Sigma_L0 = g_omg_L0[i]*omega[i] + g_phi_L0[i]*phi[i]
        Sigma_Sp = g_omg_Sp[i]*omega[i] - g_rho_Sp[i]*rho[i]
        Sigma_S0 = g_omg_S0[i]*omega[i]
        Sigma_Sm = g_omg_Sm[i]*omega[i] + g_rho_Sm[i]*rho[i]
        Sigma_X0 = g_omg_X0[i]*omega[i] - g_rho_X0[i]*rho[i]
        Sigma_Xm = g_omg_Xm[i]*omega[i] + g_rho_Xm[i]*rho[i]
        #rearrangemet terms
        Sigma_r_NN  = (dg_omg_NN[i]/g_omg_NN[i])*(m_m[1]*omega[i])**2
        Sigma_r_NN += (dg_rho_NN[i]/g_rho_NN[i])*(m_m[2]*rho[i]  )**2
        Sigma_r_NN -= (dg_sig_NN[i]/g_sig_NN[i])*(m_m[0]*sigma[i])**2
        #Lambda
        Sigma_r_L0  = (dg_omg_L0[i]/g_omg_L0[i])*(m_m[1]*omega[i])**2
        Sigma_r_L0 -= (dg_sig_L0[i]/g_sig_L0[i])*(m_m[0]*sigma[i])**2
        Sigma_r_L0 -= (dg_sgs_L0[i]/g_sgs_L0[i])*(m_m[3]*sig_s[i])**2
        Sigma_r_L0 += (dg_phi_L0[i]/g_phi_L0[i])*(m_m[4]*phi[i]  )**2
        #Sigma plus
        Sigma_r_Sp  = (dg_omg_Sp[i]/g_omg_Sp[i])*(m_m[1]*omega[i])**2
        Sigma_r_Sp += (dg_rho_Sp[i]/g_rho_Sp[i])*(m_m[2]*rho[i]  )**2
        Sigma_r_Sp -= (dg_sig_Sp[i]/g_sig_Sp[i])*(m_m[0]*sigma[i])**2
        #Sigma 0
        Sigma_r_S0  = (dg_omg_S0[i]/g_omg_S0[i])*(m_m[1]*omega[i])**2
        Sigma_r_S0 -= (dg_sig_S0[i]/g_sig_S0[i])*(m_m[0]*sigma[i])**2
        #Sigma minus
        Sigma_r_Sm  = (dg_omg_Sm[i]/g_omg_Sm[i])*(m_m[1]*omega[i])**2
        Sigma_r_Sm += (dg_rho_Sm[i]/g_rho_Sm[i])*(m_m[2]*rho[i]  )**2
        Sigma_r_Sm -= (dg_sig_Sm[i]/g_sig_Sm[i])*(m_m[0]*sigma[i])**2
        #Xi 0
        Sigma_r_X0  = (dg_omg_X0[i]/g_omg_X0[i])*(m_m[1]*omega[i])**2
        Sigma_r_X0 += (dg_rho_X0[i]/g_rho_X0[i])*(m_m[2]*rho[i]  )**2
        Sigma_r_X0 -= (dg_sig_X0[i]/g_sig_X0[i])*(m_m[0]*sigma[i])**2
        #Xi minus
        Sigma_r_Xm  = (dg_omg_Xm[i]/g_omg_Xm[i])*(m_m[1]*omega[i])**2
        Sigma_r_Xm += (dg_rho_Xm[i]/g_rho_Xm[i])*(m_m[2]*rho[i]  )**2
        Sigma_r_Xm -= (dg_sig_Xm[i]/g_sig_Xm[i])*(m_m[0]*sigma[i])**2
        #Total baryon energy
        tot_ene  = Inte(ene, (m_eff_p,  Sigma_Np + Sigma_r_NN, mu_pr[i], T[i]))
        tot_ene += Inte(ene, (m_eff_n,  Sigma_Nn + Sigma_r_NN, mu_ne[i], T[i]))
        tot_ene += Inte(ene, (m_eff_L0, Sigma_L0 + Sigma_r_L0, mu_L0[i], T[i]))
        tot_ene += Inte(ene, (m_eff_Sp, Sigma_Sp + Sigma_r_Sp, mu_Sp[i], T[i]))
        tot_ene += Inte(ene, (m_eff_S0, Sigma_S0 + Sigma_r_S0, mu_S0[i], T[i]))
        tot_ene += Inte(ene, (m_eff_Sm, Sigma_Sm + Sigma_r_Sm, mu_Sm[i], T[i]))
        tot_ene += Inte(ene, (m_eff_X0, Sigma_X0 + Sigma_r_X0, mu_X0[i], T[i]))
        tot_ene += Inte(ene, (m_eff_Xm, Sigma_Xm + Sigma_r_Xm, mu_Xm[i], T[i]))
        #leptons energy
        tot_ene += Inte(ene, (m_l[0], 0, mu_el[i], T[i]))
        tot_ene += Inte(ene, (m_l[1], 0, mu_mu[i], T[i]))
        tot_ene += Inte(ene, (m_l[2], 0, mu_nu[i], T[i]))
        #mesons energy
        ene_m  = 0.5*(m_m[0]**2*sigma[i]**2 + m_m[1]**2*omega[i]**2 + m_m[2]**2*rho[i]**2)
        ene_m += 0.5*(m_m[3]**2*sig_s[i]**2 + m_m[4]**2*phi[i]**2)
        tot_ene += ene_m

        #system's total energy
        energ[i] =  tot_ene

    return energ


def total_pressure(nb_dens, sigma, omega, rho, phi, sig_s,
                   n_pro, n_neu, n_ele, n_muo, n_nuu, n_lam,
                   n_Sgp, n_Sg0, n_Sgm, n_Xi0, n_Xim,
                   mu_pr, mu_ne, mu_L0, mu_Sp, mu_S0, mu_Sm,
                   mu_X0, mu_Xm, mu_el, mu_mu, mu_nu, T,
                   g_sig_NN,  g_omg_NN,  g_rho_NN,
                   g_sig_L0,  g_omg_L0,  g_phi_L0, g_sgs_L0,
                   g_sig_Sp,  g_omg_Sp,  g_rho_Sp,
                   g_sig_S0,  g_omg_S0,
                   g_sig_Sm,  g_omg_Sm,  g_rho_Sm,
                   g_sig_X0,  g_omg_X0,  g_rho_X0,
                   g_sig_Xm,  g_omg_Xm,  g_rho_Xm,
                   dg_sig_NN, dg_omg_NN, dg_rho_NN,
                   dg_sig_L0, dg_omg_L0, dg_phi_L0, dg_sgs_L0,
                   dg_sig_Sp, dg_omg_Sp, dg_rho_Sp,
                   dg_sig_S0, dg_omg_S0,
                   dg_sig_Sm, dg_omg_Sm, dg_rho_Sm,
                   dg_sig_X0, dg_omg_X0, dg_rho_X0,
                   dg_sig_Xm, dg_omg_Xm, dg_rho_Xm
                  ):
    '''
    Computation of the system's total pressure

    Parameters
    ----------
    nb_dens : 1darray
        total baryonic density
    sigma, omega, rho, sig_s, phi : 1darray
        array for the mesons solutions
    mu_pr, ..., mu_mu : 1darrray
        array with chemical potential for each species
    T : 1darray
        temperature
    g_sig_NN, ..., dg_rho_Xm : 1darray
        array with the value of coupling constant

    Retunrs
    -------
    press : 1darray
        array for the total pressure
    '''
    number_dens = len(nb_dens)
    #array for the energy
    press = np.zeros(number_dens)

    for i, nb in enumerate(nb_dens):

        #effective masses
        m_eff_p  = m_n[0] - sigma[i]*g_sig_NN[i]
        m_eff_n  = m_n[1] - sigma[i]*g_sig_NN[i]
        m_eff_L0 = m_n[2] - sigma[i]*g_sig_L0[i] - sig_s[i]*g_sgs_L0[i]
        m_eff_Sp = m_n[3] - sigma[i]*g_sig_Sp[i]
        m_eff_S0 = m_n[4] - sigma[i]*g_sig_S0[i]
        m_eff_Sm = m_n[5] - sigma[i]*g_sig_Sm[i]
        m_eff_X0 = m_n[6] - sigma[i]*g_sig_X0[i]
        m_eff_Xm = m_n[7] - sigma[i]*g_sig_Xm[i]
        #energy terms for baryons
        Sigma_Np = g_omg_NN[i]*omega[i] - g_rho_NN[i]*rho[i]
        Sigma_Nn = g_omg_NN[i]*omega[i] + g_rho_NN[i]*rho[i]
        Sigma_L0 = g_omg_L0[i]*omega[i] + g_phi_L0[i]*phi[i]
        Sigma_Sp = g_omg_Sp[i]*omega[i] - g_rho_Sp[i]*rho[i]
        Sigma_S0 = g_omg_S0[i]*omega[i]
        Sigma_Sm = g_omg_Sm[i]*omega[i] + g_rho_Sm[i]*rho[i]
        Sigma_X0 = g_omg_X0[i]*omega[i] - g_rho_X0[i]*rho[i]
        Sigma_Xm = g_omg_Xm[i]*omega[i] + g_rho_Xm[i]*rho[i]
        #rearrangemet terms
        Sigma_r_NN  = (dg_omg_NN[i]/g_omg_NN[i])*(m_m[1]*omega[i])**2
        Sigma_r_NN += (dg_rho_NN[i]/g_rho_NN[i])*(m_m[2]*rho[i]  )**2
        Sigma_r_NN -= (dg_sig_NN[i]/g_sig_NN[i])*(m_m[0]*sigma[i])**2
        #Lambda
        Sigma_r_L0  = (dg_omg_L0[i]/g_omg_L0[i])*(m_m[1]*omega[i])**2
        Sigma_r_L0 -= (dg_sig_L0[i]/g_sig_L0[i])*(m_m[0]*sigma[i])**2
        Sigma_r_L0 -= (dg_sgs_L0[i]/g_sgs_L0[i])*(m_m[3]*sig_s[i])**2
        Sigma_r_L0 += (dg_phi_L0[i]/g_phi_L0[i])*(m_m[4]*phi[i]  )**2
        #Sigma plus
        Sigma_r_Sp  = (dg_omg_Sp[i]/g_omg_Sp[i])*(m_m[1]*omega[i])**2
        Sigma_r_Sp += (dg_rho_Sp[i]/g_rho_Sp[i])*(m_m[2]*rho[i]  )**2
        Sigma_r_Sp -= (dg_sig_Sp[i]/g_sig_Sp[i])*(m_m[0]*sigma[i])**2
        #Sigma 0
        Sigma_r_S0  = (dg_omg_S0[i]/g_omg_S0[i])*(m_m[1]*omega[i])**2
        Sigma_r_S0 -= (dg_sig_S0[i]/g_sig_S0[i])*(m_m[0]*sigma[i])**2
        #Sigma minus
        Sigma_r_Sm  = (dg_omg_Sm[i]/g_omg_Sm[i])*(m_m[1]*omega[i])**2
        Sigma_r_Sm += (dg_rho_Sm[i]/g_rho_Sm[i])*(m_m[2]*rho[i]  )**2
        Sigma_r_Sm -= (dg_sig_Sm[i]/g_sig_Sm[i])*(m_m[0]*sigma[i])**2
        #Xi 0
        Sigma_r_X0  = (dg_omg_X0[i]/g_omg_X0[i])*(m_m[1]*omega[i])**2
        Sigma_r_X0 += (dg_rho_X0[i]/g_rho_X0[i])*(m_m[2]*rho[i]  )**2
        Sigma_r_X0 -= (dg_sig_X0[i]/g_sig_X0[i])*(m_m[0]*sigma[i])**2
        #Xi minus
        Sigma_r_Xm  = (dg_omg_Xm[i]/g_omg_Xm[i])*(m_m[1]*omega[i])**2
        Sigma_r_Xm += (dg_rho_Xm[i]/g_rho_Xm[i])*(m_m[2]*rho[i]  )**2
        Sigma_r_Xm -= (dg_sig_Xm[i]/g_sig_Xm[i])*(m_m[0]*sigma[i])**2
        #Total baryon pressure
        tot_pre  = Inte(pre, (m_eff_p,  Sigma_Np + Sigma_r_NN, mu_pr[i], T[i]))/3
        tot_pre += Inte(pre, (m_eff_n,  Sigma_Nn + Sigma_r_NN, mu_ne[i], T[i]))/3
        tot_pre += Inte(pre, (m_eff_L0, Sigma_L0 + Sigma_r_L0, mu_L0[i], T[i]))/3
        tot_pre += Inte(pre, (m_eff_Sp, Sigma_Sp + Sigma_r_Sp, mu_Sp[i], T[i]))/3
        tot_pre += Inte(pre, (m_eff_S0, Sigma_S0 + Sigma_r_S0, mu_S0[i], T[i]))/3
        tot_pre += Inte(pre, (m_eff_Sm, Sigma_Sm + Sigma_r_Sm, mu_Sm[i], T[i]))/3
        tot_pre += Inte(pre, (m_eff_X0, Sigma_X0 + Sigma_r_X0, mu_X0[i], T[i]))/3
        tot_pre += Inte(pre, (m_eff_Xm, Sigma_Xm + Sigma_r_Xm, mu_Xm[i], T[i]))/3
        #leptons pressure
        tot_pre += Inte(pre, (m_l[0], 0, mu_el[i], T[i]))/3
        tot_pre += Inte(pre, (m_l[1], 0, mu_mu[i], T[i]))/3
        tot_pre += Inte(pre, (m_l[2], 0, mu_nu[i], T[i]))/3
        #mesons pressure
        pre_m  = 0.5*(-m_m[0]**2*sigma[i]**2 + m_m[1]**2*omega[i]**2 + m_m[2]**2*rho[i]**2)
        pre_m += 0.5*(-m_m[3]**2*sig_s[i]**2 + m_m[4]**2*phi[i]**2)
        tot_pre += pre_m
        #rearrangement term
        pre_r  = Sigma_r_NN*(n_pro[i]+n_neu[i]) + Sigma_r_L0*n_lam[i] + Sigma_r_Sp*n_Sgp[i]
        pre_r += Sigma_r_S0*n_Sg0[i] + Sigma_r_Sm*n_Sgm[i] + Sigma_r_X0*n_Xi0[i] + Sigma_r_Xm*n_Xim[i]
        tot_pre += pre_r
        #system's total pressure
        press[i] =  tot_pre

    return press

#==================================================================
# density-dependet coupling-constant
#==================================================================

#constants as the density varies for nucleon
g_sig_NN = g_sat_NN[0] * f(nb_dens, 0, const=const_nuc)
g_omg_NN = g_sat_NN[1] * f(nb_dens, 1, const=const_nuc)
g_rho_NN = g_sat_NN[2] * g(nb_dens,    const=const_nuc)
#constants as the density varies for Lambda
g_sig_L0 = g_sat_L0[0] * f(nb_dens, 0, const=const_hyp)
g_omg_L0 = g_sat_L0[1] * f(nb_dens, 1, const=const_hyp)
g_phi_L0 = g_sat_L0[2] * f(nb_dens, 1, const=const_hyp)
g_sgs_L0 = g_sat_L0[3] * f(nb_dens, 0, const=const_hyp)
#constants as the density varies for Sigma plus
g_sig_Sp = g_sat_Sp[0] * f(nb_dens, 0, const=const_hyp)
g_omg_Sp = g_sat_Sp[1] * f(nb_dens, 1, const=const_hyp)
g_rho_Sp = g_sat_Sp[2] * g(nb_dens,    const=const_hyp)
#constants as the density varies for Sigma 0
g_sig_S0 = g_sat_S0[0] * f(nb_dens, 0, const=const_hyp)
g_omg_S0 = g_sat_S0[1] * f(nb_dens, 1, const=const_hyp)
#constants as the density varies for Sigma minus
g_sig_Sm = g_sat_Sm[0] * f(nb_dens, 0, const=const_hyp)
g_omg_Sm = g_sat_Sm[1] * f(nb_dens, 1, const=const_hyp)
g_rho_Sm = g_sat_Sm[2] * g(nb_dens,    const=const_hyp)
#constants as the density varies for Xi 0
g_sig_X0 = g_sat_X0[0] * f(nb_dens, 0, const=const_hyp)
g_omg_X0 = g_sat_X0[1] * f(nb_dens, 1, const=const_hyp)
g_rho_X0 = g_sat_X0[2] * g(nb_dens,    const=const_hyp)
#constants as the density varies for Xi minus
g_sig_Xm = g_sat_Xm[0] * f(nb_dens, 0, const=const_hyp)
g_omg_Xm = g_sat_Xm[1] * f(nb_dens, 1, const=const_hyp)
g_rho_Xm = g_sat_Xm[2] * g(nb_dens,    const=const_hyp)
#constants derivatives as the density varies for nucleons
dg_sig_NN = g_sat_NN[0] * f(nb_dens, 0, const=const_nuc, derive=True)
dg_omg_NN = g_sat_NN[1] * f(nb_dens, 1, const=const_nuc, derive=True)
dg_rho_NN = g_sat_NN[2] * g(nb_dens,    const=const_nuc, derive=True)
#constants derivatives as the density varies for Lambda
dg_sig_L0 = g_sat_L0[0] * f(nb_dens, 0, const=const_hyp, derive=True)
dg_omg_L0 = g_sat_L0[1] * f(nb_dens, 1, const=const_hyp, derive=True)
dg_phi_L0 = g_sat_L0[2] * f(nb_dens, 1, const=const_hyp, derive=True)
dg_sgs_L0 = g_sat_L0[3] * f(nb_dens, 0, const=const_hyp, derive=True)
#constants derivatives as the density varies for Sigma plus
dg_sig_Sp = g_sat_Sp[0] * f(nb_dens, 0, const=const_hyp, derive=True)
dg_omg_Sp = g_sat_Sp[1] * f(nb_dens, 1, const=const_hyp, derive=True)
dg_rho_Sp = g_sat_Sp[2] * g(nb_dens,    const=const_hyp, derive=True)
#constants derivatives as the density varies for Sigma 0
dg_sig_S0 = g_sat_S0[0] * f(nb_dens, 0, const=const_hyp, derive=True)
dg_omg_S0 = g_sat_S0[1] * f(nb_dens, 1, const=const_hyp, derive=True)
#constants derivatives as the density varies for Sigma minus
dg_sig_Sm = g_sat_Sm[0] * f(nb_dens, 0, const=const_hyp, derive=True)
dg_omg_Sm = g_sat_Sm[1] * f(nb_dens, 1, const=const_hyp, derive=True)
dg_rho_Sm = g_sat_Sm[2] * g(nb_dens,    const=const_hyp, derive=True)
#constants derivatives as the density varies for Xi 0
dg_sig_X0 = g_sat_X0[0] * f(nb_dens, 0, const=const_hyp, derive=True)
dg_omg_X0 = g_sat_X0[1] * f(nb_dens, 1, const=const_hyp, derive=True)
dg_rho_X0 = g_sat_X0[2] * g(nb_dens,    const=const_hyp, derive=True)
#constants derivatives as the density varies for Xi minus
dg_sig_Xm = g_sat_Xm[0] * f(nb_dens, 0, const=const_hyp, derive=True)
dg_omg_Xm = g_sat_Xm[1] * f(nb_dens, 1, const=const_hyp, derive=True)
dg_rho_Xm = g_sat_Xm[2] * g(nb_dens,    const=const_hyp, derive=True)


#==================================================================
# Loop over densities and compute solution
#==================================================================

t0 = time.time()

print("Solving system...")
for i, nb in enumerate(nb_dens):

    #starting point depending on i to speed up the computation
    start = (sigma[i], omega[i], rho[i],   n_pro[i], n_neu[i], n_ele[i],
             mu_pr[i], mu_ne[i], mu_el[i], n_muo[i], mu_mu[i], sig_s[i],
             phi[i],   n_lam[i], mu_L0[i], n_Sgp[i], n_Sg0[i], n_Sgm[i],
             mu_Sp[i], mu_S0[i], mu_Sm[i], n_Xi0[i], n_Xim[i], mu_Xm[i],
             mu_X0[i], T_nbd[i], n_nuu[i], mu_nu[i])

    #we solve the system and keep the solution
    arg = (nb,
           g_sig_NN[i],  g_omg_NN[i],  g_rho_NN[i], dg_sig_NN[i], dg_omg_NN[i], dg_rho_NN[i],
           S,
           g_sig_L0[i],  g_omg_L0[i],  g_phi_L0[i],  g_sgs_L0[i], dg_sig_L0[i], dg_omg_L0[i],
           dg_phi_L0[i], dg_sgs_L0[i], g_sig_Sp[i],  g_omg_Sp[i],  g_rho_Sp[i], dg_sig_Sp[i],
           dg_omg_Sp[i], dg_rho_Sp[i], g_sig_S0[i],  g_omg_S0[i], dg_sig_S0[i], dg_omg_S0[i],
           g_sig_Sm[i],  g_omg_Sm[i],  g_rho_Sm[i], dg_sig_Sm[i], dg_omg_Sm[i], dg_rho_Sm[i],
           g_sig_X0[i],  g_omg_X0[i],  g_rho_X0[i], dg_sig_X0[i], dg_omg_X0[i], dg_rho_X0[i],
           g_sig_Xm[i],  g_omg_Xm[i],  g_rho_Xm[i], dg_sig_Xm[i], dg_omg_Xm[i], dg_rho_Xm[i])

    sol = root(system , start, args=arg, tol=1e-10, method='hybr')
    #save solution
    sigma[i+1], omega[i+1], rho[i+1]   = sol.x[0:3]
    n_pro[i+1], n_neu[i+1], n_ele[i+1] = sol.x[3:6]
    mu_pr[i+1], mu_ne[i+1], mu_el[i+1] = sol.x[6:9]
    n_muo[i+1], mu_mu[i+1]             = sol.x[9:11]
    sig_s[i+1], phi[i+1]               = sol.x[11:13]
    n_lam[i+1], mu_L0[i+1]             = sol.x[13:15]
    n_Sgp[i+1], n_Sg0[i+1], n_Sgm[i+1] = sol.x[15:18]
    mu_Sp[i+1], mu_S0[i+1], mu_Sm[i+1] = sol.x[18:21]
    n_Xi0[i+1], n_Xim[i+1]             = sol.x[21:23]
    mu_X0[i+1], mu_Xm[i+1]             = sol.x[23:25]
    T_nbd[i+1]                         = sol.x[25]
    n_nuu[i+1], mu_nu[i+1]             = sol.x[26:28]

    # To avoid numerical problem if numerical density became very low
    n_ele[i+1] = 0 if n_ele[i+1]/nb <1e-6 else n_ele[i+1]
    n_muo[i+1] = 0 if n_muo[i+1]/nb <1e-6 else n_muo[i+1]
    n_lam[i+1] = 0 if n_lam[i+1]/nb <1e-6 else n_lam[i+1]
    n_Sgp[i+1] = 0 if n_Sgp[i+1]/nb <1e-6 else n_Sgp[i+1]
    n_Sg0[i+1] = 0 if n_Sg0[i+1]/nb <1e-6 else n_Sg0[i+1]
    n_Sgm[i+1] = 0 if n_Sgm[i+1]/nb <1e-6 else n_Sgm[i+1]
    n_Xi0[i+1] = 0 if n_Xi0[i+1]/nb <1e-6 else n_Xi0[i+1]
    n_Xim[i+1] = 0 if n_Xim[i+1]/nb <1e-6 else n_Xim[i+1]

    print(f"{(i+1)/(number_dens) * 100:.1f} % \r", end='')



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
n_nuu = n_nuu[1:]
n_lam = n_lam[1:]
n_Sgp = n_Sgp[1:]
n_Sg0 = n_Sg0[1:]
n_Sgm = n_Sgm[1:]
n_Xi0 = n_Xi0[1:]
n_Xim = n_Xim[1:]
mu_pr = mu_pr[1:]
mu_ne = mu_ne[1:]
mu_L0 = mu_L0[1:]
mu_Sp = mu_Sp[1:]
mu_S0 = mu_S0[1:]
mu_Sm = mu_Sm[1:]
mu_X0 = mu_X0[1:]
mu_Xm = mu_Xm[1:]
mu_el = mu_el[1:]
mu_mu = mu_mu[1:]
mu_nu = mu_nu[1:]
T_nbd = T_nbd[1:]


#==================================================================
## Computing termodinamical quantities
#==================================================================

print("Computing termodinamical quantities...")
#energy and pressure computation
ene_nb = total_energy(nb_dens, sigma, omega, rho, phi, sig_s,
                      mu_pr, mu_ne, mu_L0, mu_Sp, mu_S0, mu_Sm,
                      mu_X0, mu_Xm, mu_el, mu_mu, mu_nu, T_nbd,
                      g_sig_NN,  g_omg_NN,  g_rho_NN,
                      g_sig_L0,  g_omg_L0,  g_phi_L0, g_sgs_L0,
                      g_sig_Sp,  g_omg_Sp,  g_rho_Sp,
                      g_sig_S0,  g_omg_S0,
                      g_sig_Sm,  g_omg_Sm,  g_rho_Sm,
                      g_sig_X0,  g_omg_X0,  g_rho_X0,
                      g_sig_Xm,  g_omg_Xm,  g_rho_Xm,
                      dg_sig_NN, dg_omg_NN, dg_rho_NN,
                      dg_sig_L0, dg_omg_L0, dg_phi_L0, dg_sgs_L0,
                      dg_sig_Sp, dg_omg_Sp, dg_rho_Sp,
                      dg_sig_S0, dg_omg_S0,
                      dg_sig_Sm, dg_omg_Sm, dg_rho_Sm,
                      dg_sig_X0, dg_omg_X0, dg_rho_X0,
                      dg_sig_Xm, dg_omg_Xm, dg_rho_Xm
                      )

print("Energy done")

pre_nb = total_pressure(nb_dens, sigma, omega, rho, phi, sig_s,
                        n_pro, n_neu, n_ele, n_muo, n_nuu, n_lam,
                        n_Sgp, n_Sg0, n_Sgm, n_Xi0, n_Xim,
                        mu_pr, mu_ne, mu_L0, mu_Sp, mu_S0, mu_Sm,
                        mu_X0, mu_Xm, mu_el, mu_mu, mu_nu, T_nbd,
                        g_sig_NN,  g_omg_NN,  g_rho_NN,
                        g_sig_L0,  g_omg_L0,  g_phi_L0, g_sgs_L0,
                        g_sig_Sp,  g_omg_Sp,  g_rho_Sp,
                        g_sig_S0,  g_omg_S0,
                        g_sig_Sm,  g_omg_Sm,  g_rho_Sm,
                        g_sig_X0,  g_omg_X0,  g_rho_X0,
                        g_sig_Xm,  g_omg_Xm,  g_rho_Xm,
                        dg_sig_NN, dg_omg_NN, dg_rho_NN,
                        dg_sig_L0, dg_omg_L0, dg_phi_L0, dg_sgs_L0,
                        dg_sig_Sp, dg_omg_Sp, dg_rho_Sp,
                        dg_sig_S0, dg_omg_S0,
                        dg_sig_Sm, dg_omg_Sm, dg_rho_Sm,
                        dg_sig_X0, dg_omg_X0, dg_rho_X0,
                        dg_sig_Xm, dg_omg_Xm, dg_rho_Xm
                       )

print("Pressure done")

T_nbd  *= hbc
ene_nb *= hbc
pre_nb *= hbc

dt = time.time() - t0            #end of computation
print(f'Elapsed time: {dt}')

#==================================================================
## Plot
#==================================================================

#abundances plot
plt.figure(1)
plt.title(f'Abundance of each species at S/A = {S}', fontsize=15)
plt.xlabel('$n_B$ [fm$^{-3}$]', fontsize=15)
plt.ylabel('$Y_i$', fontsize=15)
plt.plot(nb_dens, n_pro/nb_dens, 'red',     label='$n_p/n_b$')
plt.plot(nb_dens, n_neu/nb_dens, 'black',   label='$n_n/n_b$')
plt.plot(nb_dens, n_ele/nb_dens, 'blue',    label='$n_e/n_b$')
plt.plot(nb_dens, n_muo/nb_dens, 'purple',  label='$n_{\mu}/n_b$')
plt.plot(nb_dens, n_lam/nb_dens, 'green',   label='$n_{\Lambda}/n_b$')
plt.plot(nb_dens, n_Sgp/nb_dens, 'orange',  label='$n_{\Sigma^+}/n_b$')
plt.plot(nb_dens, n_Sg0/nb_dens, 'lime',    label='$n_{\Sigma^0}/n_b$')
plt.plot(nb_dens, n_Sgm/nb_dens, 'pink',    label='$n_{\Sigma^-}/n_b$')
plt.plot(nb_dens, n_Xi0/nb_dens, 'navy',    label='$n_{\Xi^0}/n_b$')
plt.plot(nb_dens, n_Xim/nb_dens, 'fuchsia', label='$n_{\Xi^-}/n_b$')
plt.plot(nb_dens, n_nuu/nb_dens, 'deepskyblue', label=r'$n_{\nu}/n_b$')
plt.legend(loc='best', ncol=2)
plt.yscale('log')
plt.grid()
#plt.savefig("Abbondanze_T10")

#chemical potential plot
plt.figure(2)
plt.title(f'chemical potential at S = {S}', fontsize=15)
plt.ylabel('chemical potential [$fm^{-1}$]', fontsize=15)
plt.xlabel('density [$fm^{-3}$]', fontsize=15)
plt.grid()
plt.plot(nb_dens, mu_ne, 'k', label='$\mu_n$')
plt.plot(nb_dens, mu_pr, 'red', label='$\mu_p$')
plt.plot(nb_dens, mu_el, 'blue',label='$\mu_e$')
plt.plot(nb_dens, mu_mu, 'purple', label='$\mu_{\mu}$')
plt.plot(nb_dens, mu_L0, 'green',   label='$\mu_{\Lambda}$')
plt.plot(nb_dens, mu_Sp, 'orange',  label='$\mu_{\Sigma^+}$')
plt.plot(nb_dens, mu_S0, 'lime',    label='$\mu_{\Sigma^0}$')
plt.plot(nb_dens, mu_Sm, 'pink',    label='$\mu_{\Sigma^-}$')
plt.plot(nb_dens, mu_X0, 'navy',    label='$\mu_{\Xi^0}$')
plt.plot(nb_dens, mu_Xm, 'fuchsia', label='$\mu_{\Xi^-}$')
plt.plot(nb_dens, mu_nu, 'deepskyblue', label=r'$\mu_{\nu}$')
plt.legend(loc='best')
#plt.savefig("pot_chimici_T10")

#energy plot
plt.figure(3)
plt.title(f'Energy at S = {S}', fontsize=15)
plt.ylabel(r'$\varepsilon$ [Mev/$fm^{-3}$]')
plt.xlabel('density [fm$^{-3}$]', fontsize=15)
plt.grid()
plt.plot(nb_dens, ene_nb, 'k')
#plt.savefig("Energia_T10")

#pressure plot
plt.figure(4)
plt.title(f'Pressure at S = {S}', fontsize=15)
plt.ylabel('P [Mev/$fm^3$]', fontsize=15)
plt.xlabel('density [fm$^{-3}$]', fontsize=15)
plt.grid()
plt.plot(nb_dens, pre_nb, 'k')
#plt.savefig("Pressione_T10")

#temperature plot
plt.figure(5)
plt.title(f'Temperature at S = {S}', fontsize=15)
plt.ylabel('T [MeV]', fontsize=15)
plt.xlabel('density [fm$^{-3}$]', fontsize=15)
plt.grid()
plt.plot(nb_dens, T_nbd, 'k')
#plt.savefig("Temperatura_S3")

plt.show()

#==================================================================
# Save data
#==================================================================

if SAVE :

    file = open(path1, "w")
    file.write("# nb [fm^-3] \t Yp \t Yn \t Ye \t Ym \t Ynu \t YL \t YSp \t YS0 \t YSm \t YX0 \t YXm \n")

    List = zip(nb_dens, n_pro, n_neu, n_ele, n_muo, n_nuu, n_lam, n_Sgp, n_Sg0, n_Sgm, n_Xi0, n_Xim)

    for nb, p, n, e, m, nu, L, Sp, S0, Sm, X0, Xm in List:
        data1 = f"{nb} \t {p/nb} \t {n/nb} \t {e/nb} \t {m/nb} \t {nu/nb} \t {L/nb}"
        data2 = f"{Sp/nb} \t {S0/nb} \t {Sm/nb} \t {X0/nb} \t {Xm/nb} \n"
        file.write(f"{data1} \t {data2}")

    file.close()

    #======================================

    file = open(path2, "w")
    str1 = f'# nb [fm^-3] \t mu_n [fm^-1] \t mu_p [fm^-1] \t mu_e [fm^-1] \t mu_m [fm^-1] \t mu_nu [fm^-1] \t'
    str2 = f'  mu_L0 [fm^-1] \t mu_Sp [fm^-1] \t mu_S0 [fm^-1] \t mu_Sm [fm^-1] \t mu_X0 [fm^-1] \t mu_Xm [fm^-1] \n'
    file.write(f'{str1}{str2}')

    List = zip(nb_dens, mu_ne, mu_pr, mu_el, mu_mu, mu_nu, mu_L0, mu_Sp, mu_S0, mu_Sm, mu_X0, mu_Xm)

    for nb, n, p, e, m, nu, L0, Sp, S0, Sm, X0, Xm in List:
        D = f"{nb} \t {n} \t {p} \t {e} \t {m} \t{nu} \t {L0} \t {Sp} \t {S0} \t {Sm} \t {X0} \t {Xm} \n"
        file.write(D)

    file.close()

    #======================================

    file = open(path3, "w")
    file.write("# E [MeV fm^-3] \t P [MeV fm^-3] \t T [MeV] \t nb [fm^-3] \n")
    for nb, ene, pre, temp in zip(nb_dens, ene_nb, pre_nb, T_nbd):
        file.write(f"{ene} \t{pre} \t {temp} \t {nb} \n")

    file.close()
