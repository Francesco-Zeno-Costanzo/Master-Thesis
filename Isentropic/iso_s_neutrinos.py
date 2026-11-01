"""
This code solves the equations of the relativistic mean field model.
The calculation are done at constant entropy per baryon S/A = const.
The particles considered are: neutrons, protons, muons, electrons and neutrinos
"""

import time
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from scipy.optimize import root

#==================================================================
# Computation parameters
#==================================================================

number_dens = 100
min_dens    = 0.1
max_dens    = 1

nb_dens = np.logspace(np.log10(min_dens), np.log10(max_dens), number_dens)

#entropy per baryon in unit of K_B
S = 3

#==================================================================
# Path of file for saving data
#==================================================================

SAVE  = False # If False the data are not saved
path1 = fr"abb_NUCn_S{S}_dd.txt"
path2 = fr"pot_NUCn_S{S}_dd.txt"
path3 = fr"EOS_NUCn_S{S}_dd.txt"

#==================================================================
# Physical constant and coupling constant
#==================================================================

#\hbar/c in MeV fermi
hbc = 197.327

#saturation numerical densitity
nb_sat = 0.152

#baryon masses
mass_p = 938.27203
mass_n = 939.56536
m_n = np.array([mass_p, mass_n])/hbc

#lepton masses
m_el = 0.51099907
m_mu = 105.6583755
m_nu = 0
m_l = np.array([m_el, m_mu, m_nu])/hbc

#meson masses
m_sg = 550.1238
m_om = 783.0000
m_rh = 763.0000
m_m = np.array([m_sg, m_om, m_rh])/hbc

#coupling constants at nb_sat
g_sg_sat = 10.5396
g_om_sat = 13.0189
g_rh_sat = 3.6836 #we already considered 1/2
g_sat = np.array([g_sg_sat, g_om_sat, g_rh_sat])

#array to model the dependence on the coupling constants
#in order     sigma,  omega,  rho
a = np.array([1.3881, 1.3892, 0.5647])
b = np.array([1.0943, 0.9240])
c = np.array([1.7057, 1.4620])
d = np.array([0.4421, 0.4775])

#==================================================================
# Initial Value: TO CHOOSE CAREFULLY
#==================================================================

#array for the solutions
sigma = np.zeros(number_dens + 1)
omega = np.zeros(number_dens + 1)
rho   = np.zeros(number_dens + 1)
mu_pr = np.zeros(number_dens + 1)
mu_ne = np.zeros(number_dens + 1)
mu_el = np.zeros(number_dens + 1)
mu_mu = np.zeros(number_dens + 1)
mu_nu = np.zeros(number_dens + 1)
n_pro = np.zeros(number_dens + 1)
n_neu = np.zeros(number_dens + 1)
n_ele = np.zeros(number_dens + 1)
n_muo = np.zeros(number_dens + 1)
n_nuu = np.zeros(number_dens + 1)
T_nbd = np.zeros(number_dens + 1)


sigma[0], omega[0], rho[0] = g_sat/m_m**2  * min_dens#/10
mu_pr[0] = m_n[0]
mu_ne[0] = m_n[1]
mu_el[0] = m_l[0]
mu_mu[0] = m_l[1]
mu_nu[0] = m_l[2]
n_neu[0] = 0.9 * min_dens#/10
n_pro[0] = 0.1 * min_dens#/10
n_ele[0] = 0.1 * min_dens#/10
n_muo[0] = 0.0 #* min_dens
n_nuu[0] = 0.0 #* min_dens
T_nbd[0] = 14 / hbc

#==================================================================
# Auxiliary Function
#==================================================================

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
        g_n = g_sat[i]*a[i] * num/den
        return g_n

    else :
        num = 2*a[i]*(b[i] - c[i])*nb_sat**2 * (n + d[i]*nb_sat)
        den = (nb_sat**2 + c[i]*(n + d[i]*nb_sat)**2)**2
        dg_n = g_sat[i] * num/den
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
        g_n = g_sat[2] * np.exp(-a[2]*(x-1))
        return g_n

    else :
        dg_n = - (g_sat[2]*a[2]/nb_sat) * np.exp(a[2]*(1-x))
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
    integral for the nucleons entropy

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

#==================================================================
# System to solve
#==================================================================

def system(V, nb, g_sg, g_om, g_rh, dg_sg, dg_om, dg_rh, S):
    """
    system to solve

    Parameters
    ----------
    V : 1darray
        array of the variables
    nb : float
        total baryonic density
    g_sg : float
        coupling constant for sigma at density nb = n+p
    g_om : float
        coupling constant for omega at density nb = n+p
    g_rh : float
        coupling constant for rho   at density nb = n+p
    dg_sg : float
        derivative of coupling constant for sigma at density nb = n+p
    dg_om : float
        derivative of coupling constant for omega at density nb = n+p
    dg_rh : float
        derivative of coupling constant for rho   at density nb = n+p
    S : float
        Entropy

    Returns
    -------
    list of equations to solve
    ri = 0
    """
    sig, omg, rho, p, n, e, m, nu, mu_p, mu_n, mu_e, mu_m, mu_nu, T = V

    #effective masses
    m_eff_p = m_n[0] - sig*g_sg
    m_eff_n = m_n[1] - sig*g_sg
    #termini di energia
    Sigma_p = g_om*omg - g_rh*rho
    Sigma_n = g_om*omg + g_rh*rho
    Sigma_r = (dg_om/g_om)*(m_m[1]*omg)**2 + (dg_rh/g_rh)*(m_m[2]*rho)**2 - (dg_sg/g_sg)*(m_m[0]*sig)**2
    #numerical integrals for sigma
    I_p = Inte(ns, (m_eff_p, Sigma_p + Sigma_r, mu_p, T))
    I_n = Inte(ns, (m_eff_n, Sigma_n + Sigma_r, mu_n, T))
    #fermi dirac integrals baryons
    I_f_p_p = Inte(fermi_dirac_p, (m_eff_p, Sigma_p + Sigma_r, mu_p, T, True))
    I_f_n_p = Inte(fermi_dirac_p, (m_eff_n, Sigma_n + Sigma_r, mu_n, T, True))
    I_f_p_a = Inte(fermi_dirac_a, (m_eff_p, Sigma_p + Sigma_r, mu_p, T, True))
    I_f_n_a = Inte(fermi_dirac_a, (m_eff_n, Sigma_n + Sigma_r, mu_n, T, True))
    #fermi dirac integrals for leptons
    I_f_e_p  = Inte(fermi_dirac_p, (m_l[0], 0, mu_e,  T, True))
    I_f_e_a  = Inte(fermi_dirac_a, (m_l[0], 0, mu_e,  T, True))
    I_f_m_p  = Inte(fermi_dirac_p, (m_l[1], 0, mu_m,  T, True))
    I_f_m_a  = Inte(fermi_dirac_a, (m_l[1], 0, mu_m,  T, True))
    I_f_nu_p = Inte(fermi_dirac_p, (m_l[2], 0, mu_nu, T, True))
    I_f_nu_a = Inte(fermi_dirac_a, (m_l[2], 0, mu_nu, T, True))
    #baryons entropy
    I_e_p = Inte(entropy, (m_eff_p, Sigma_p + Sigma_r, mu_p, T))
    I_e_n = Inte(entropy, (m_eff_n, Sigma_n + Sigma_r, mu_n, T))
    #lepton term
    I_e_e  = Inte(entropy, (m_l[0], 0, mu_e,  T))
    I_e_m  = Inte(entropy, (m_l[1], 0, mu_m,  T))
    I_e_nu = Inte(entropy, (m_l[2], 0, mu_nu, T))
    #equation to solve
    r1 = sig - g_sg/m_m[0]**2 * (I_p + I_n)
    r2 = omg - g_om/m_m[1]**2 * (n + p)
    r3 = rho - g_rh/m_m[2]**2 * (n - p)
    r4 = nb - n - p
    r5 = p - e - m
    r6 = p - (I_f_p_p - I_f_p_a)
    r7 = n - (I_f_n_p - I_f_n_a)
    r8 = e - (I_f_e_p - I_f_e_a)
    r9 = m - (I_f_m_p - I_f_m_a)
    r10 = nu - (I_f_nu_p - I_f_nu_a)
    r11 = mu_n - mu_p - mu_e + mu_nu
    r12 = mu_e - mu_m - mu_nu
    r13 = nu + e - 0.33*nb
    r14 = S + (I_e_p + I_e_n + I_e_e + I_e_m + I_e_nu)/nb

    return [r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14]

#==================================================================
# Function for the calculation of energy and pressure (i.e. EoS)
#==================================================================

def total_energy(nb_dens, sigma, omega, rho, mu_p, mu_n, mu_e, mu_m, mu_nu, T):
    """
    Computation of the system's total energy

    Parameters
    ----------
    nb_dens : 1darray
        total baryonic density
    sigma, omega, rho : 1darray
        array with the solutions for the mesons
    mu_p, mu_n, mu_e, mu_m, mu_nu : 1darray
        array with the solutions for the chemical potential
    T : 1darray
        temperature

    Retunrs
    -------
    energ : 1darray
        array for the total energy
    """
    number_dens = len(nb_dens)
    #array for the energy
    energ = np.zeros(number_dens)

    for i, nb in enumerate(nb_dens):
        #constants as the density varies
        g_sg = f(nb, 0)
        g_om = f(nb, 1)
        g_rh = g(nb)
        #constants derivatives as the density varies
        dg_sg = f(nb, 0, derive=True)
        dg_om = f(nb, 1, derive=True)
        dg_rh = g(nb, derive=True)
        #effective masses
        m_eff_p = m_n[0] - sigma[i]*g_sg
        m_eff_n = m_n[1] - sigma[i]*g_sg
        #energy's terms
        Sigma_p = g_om*omega[i] - g_rh*rho[i]
        Sigma_n = g_om*omega[i] + g_rh*rho[i]
        Sigma_r = (dg_om/g_om)*(m_m[1]*omega[i])**2 + (dg_rh/g_rh)*(m_m[2]*rho[i])**2 - (dg_sg/g_sg)*(m_m[0]*sigma[i])**2
        #mesons energy
        ene_m = 0.5*(m_m[0]**2*sigma[i]**2 + m_m[1]**2*omega[i]**2 + m_m[2]**2*rho[i]**2)
        #nucleons energies
        ene_p = Inte(ene, (m_eff_p, Sigma_p + Sigma_r, mu_p[i], T[i]))
        ene_n = Inte(ene, (m_eff_n, Sigma_n + Sigma_r, mu_n[i], T[i]))
        #leptons energy term
        ene_e  = Inte(ene, (m_l[0], 0, mu_e[i],  T[i]))
        ene_mu = Inte(ene, (m_l[1], 0, mu_m[i],  T[i]))
        ene_nu = Inte(ene, (m_l[2], 0, mu_nu[i], T[i]))
        #system's total energy
        energ[i] =  ene_m + ene_p + ene_n + ene_e + ene_mu + ene_nu

    return energ


def total_pressure(nb_dens, sigma, omega, rho, mu_p, mu_n, mu_e, mu_m, mu_nu, T):
    """
    Computation of the system's total pressure

    Parameters
    ----------
    nb_dens : 1darray
        total baryonic density
    sigma, omega, rho : 1darray
        array for the mesons solutions
    mu_p, mu_n, mu_e, mu_m, mu_nu : 1darray
        array with the solutions for the chemical potential
    T : 1darray
        temperature

    Retunrs
    -------
    press : 1darray
        array for the total pressure
    """
    number_dens = len(nb_dens)
    #array for the pressure
    press = np.zeros(number_dens)

    for i, nb in enumerate(nb_dens):
        #constants as the density varies
        g_sg = f(nb, 0)
        g_om = f(nb, 1)
        g_rh = g(nb)
        #effective masses
        m_eff_p = m_n[0] - sigma[i]*g_sg
        m_eff_n = m_n[1] - sigma[i]*g_sg
        #constants derivatives as the density varies
        dg_sg = f(nb, 0, derive=True)
        dg_om = f(nb, 1, derive=True)
        dg_rh = g(nb, derive=True)
        #energy's terms
        Sigma_p = g_om*omega[i] - g_rh*rho[i]
        Sigma_n = g_om*omega[i] + g_rh*rho[i]
        Sigma_r = (dg_om/g_om)*(m_m[1]*omega[i])**2 + (dg_rh/g_rh)*(m_m[2]*rho[i])**2 - (dg_sg/g_sg)*(m_m[0]*sigma[i])**2
        #nucleons pressure terms
        pre_p = Inte(pre, (m_eff_p, Sigma_p + Sigma_r, mu_p[i], T[i]))
        pre_n = Inte(pre, (m_eff_n, Sigma_n + Sigma_r, mu_n[i], T[i]))
        #leptons pressure term
        pre_e  = Inte(pre, (m_l[0], 0, mu_e[i],  T[i]))
        pre_m  = Inte(pre, (m_l[1], 0, mu_m[i],  T[i]))
        pre_nu = Inte(pre, (m_l[2], 0, mu_nu[i], T[i]))
        #mesons derivative terms
        pre_m1 = Sigma_r
        #mesons pressure term
        pre_m2 = -0.5*(m_m[0]**2*sigma[i]**2 - m_m[1]**2*omega[i]**2 - m_m[2]**2*rho[i]**2)
        #system's total pressure
        press[i] = (1/3)*(pre_p + pre_n + pre_e + pre_m + pre_nu) + nb*pre_m1 + pre_m2

    return press

#==================================================================
# MAIN PROGRAM
#==================================================================

t0 = time.time()

for i, nb in enumerate(nb_dens): #loop over densities
    #constants as the density varies
    g_sg = f(nb, 0)
    g_om = f(nb, 1)
    g_rh = g(nb)
    #constants derivatives as the density varies
    dg_sg = f(nb, 0, derive=True)
    dg_om = f(nb, 1, derive=True)
    dg_rh = g(nb, derive=True)
    #starting point depending on i to speed up the computation
    start = (sigma[i], omega[i], rho[i],
             n_pro[i], n_neu[i], n_ele[i], n_muo[i], n_nuu[i],
             mu_pr[i], mu_ne[i], mu_el[i], mu_mu[i], mu_nu[i],
             T_nbd[i])
    #we solve the system and keep the solution
    arg = (nb, g_sg, g_om, g_rh, dg_sg, dg_om, dg_rh, S)
    sol = root(system , start, args=arg, tol=1e-10, method='hybr')
    #save solution
    sigma[i+1], omega[i+1], rho[i+1]                           = sol.x[0:3]
    n_pro[i+1], n_neu[i+1], n_ele[i+1], n_muo[i+1], n_nuu[i+1] = sol.x[3:8]
    mu_pr[i+1], mu_ne[i+1], mu_el[i+1], mu_mu[i+1], mu_nu[i+1] = sol.x[8:13]
    T_nbd[i+1]                                                 = sol.x[13]

    print(f"{(i+1)/(number_dens) * 100:.1f} % \r", end='')


#we delete the initial value
sigma = sigma[1:]
omega = omega[1:]
rho   = rho[1:]
mu_pr = mu_pr[1:]
mu_ne = mu_ne[1:]
mu_el = mu_el[1:]
mu_mu = mu_mu[1:]
mu_nu = mu_nu[1:]
n_pro = n_pro[1:]
n_neu = n_neu[1:]
n_ele = n_ele[1:]
n_muo = n_muo[1:]
n_nuu = n_nuu[1:]
T_nbd = T_nbd[1:]

#energy and pressure computation
ene_nb = total_energy(  nb_dens, sigma, omega, rho, mu_pr, mu_ne, mu_el, mu_mu, mu_nu, T_nbd)
pre_nb = total_pressure(nb_dens, sigma, omega, rho, mu_pr, mu_ne, mu_el, mu_mu, mu_nu, T_nbd)

#enrgy per nucleon
ene_nb *= hbc
#pressure
pre_nb *= hbc
#temperature
T_nbd  *= hbc

dt = time.time() - t0
print(f'tempo impeigato: {dt}')


#abundances plot
plt.figure(1)
plt.title(f'Abundance of each species at S/A = {S}', fontsize=15)
plt.xlabel('$n_B$ [fm$^{-3}$]', fontsize=15)
plt.ylabel('$Y_i$', fontsize=15)
plt.plot(nb_dens, n_pro/nb_dens, 'red', label=r'$n_p/n_b$')
plt.plot(nb_dens, n_neu/nb_dens, 'black',label=r'$n_n/n_b$')
plt.plot(nb_dens, n_ele/nb_dens, 'blue',label=r'$n_e/n_b$')
plt.plot(nb_dens, n_muo/nb_dens, 'purple', label=r'$n_{\mu}/n_b$')
plt.plot(nb_dens, n_nuu/nb_dens, 'deepskyblue', label=r'$n_{\nu}/n_b$')
plt.legend(loc='best')
#plt.xscale('log')
plt.yscale('log')
plt.grid()
#plt.savefig("Abbondanze_S3")

#chemical potential plot
plt.figure(2)
plt.title(f'chemical potential for protons and neutrons at S = {S}', fontsize=15)
plt.ylabel('chemical potential [fm]', fontsize=15)
plt.xlabel('density [fm$^{-3}$]', fontsize=15)
plt.grid()
plt.plot(nb_dens, mu_ne, 'k', label=r'$\mu_n$')
plt.plot(nb_dens, mu_pr, 'red', label=r'$\mu_p$')
plt.plot(nb_dens, mu_el, 'blue',label=r'$\mu_e$')
plt.plot(nb_dens, mu_mu, 'purple', label=r'$\mu_{\mu}$')
plt.plot(nb_dens, mu_nu, 'deepskyblue', label=r'$\mu_{\nu}$')
plt.legend(loc='best')
#plt.savefig("pot_chimici_S3")

#energy plot
plt.figure(3)
plt.title(f'Energy density at S = {S}', fontsize=15)
plt.ylabel(r'$\varepsilon $ [Mev/fm$^{3}$]', fontsize=15)
plt.xlabel('density [fm$^{-3}$]', fontsize=15)
plt.grid()
plt.plot(nb_dens, ene_nb, 'k')
#plt.savefig("Energia_S3")

#pressure plot
plt.figure(4)
plt.title(f'Pressure at S = {S}', fontsize=15)
plt.ylabel('P [Mev/$fm^3$]', fontsize=15)
plt.xlabel('density [fm$^{-3}$]', fontsize=15)
plt.grid()
plt.plot(nb_dens, pre_nb, 'k')
#plt.savefig("Pressione_S3")

#Temperature plot
plt.figure(5)
plt.title(f'Temperature at S/A = {S}', fontsize=15)
plt.ylabel('T [K]', fontsize=15)
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
    file.write("# nb [fm^-3] \t Yp \t Yn \t Ye \t Ym \t Ynu \n")

    List = zip(nb_dens, n_pro, n_neu, n_ele, n_muo, n_nuu)

    for nb, p, n, e, m, nu in List:
        data1 = f"{nb} \t {p/nb} \t {n/nb} \t {e/nb} \t {m/nb} \t {nu/nb} \n"
        file.write(data1)

    file.close()

    #======================================

    file = open(path2, "w")
    str1 = f'# nb [fm^-3] \t mu_n [fm^-1] \t mu_p [fm^-1] \t mu_e [fm^-1] \t mu_m [fm^-1] \t mu_nu [fm^-1] \n'
    file.write(str1)

    List = zip(nb_dens, mu_ne, mu_pr, mu_el, mu_mu, mu_nu)

    for nb, n, p, e, m, nu in List:
        D = f"{nb} \t {n} \t {p} \t {e} \t {m} \t{nu} \n"
        file.write(D)

    file.close()

    #======================================

    file = open(path3, "w")
    file.write("# E [MeV fm^-3] \t P [MeV fm^-3] \t T [MeV] \t nb [fm^-3] \n")
    for nb, ene, pre, temp in zip(nb_dens, ene_nb, pre_nb, T_nbd):
        file.write(f"{ene} \t{pre} \t {temp} \t {nb} \n")

    file.close()

