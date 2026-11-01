"""
Code with all usefull function
"""

import numpy as np
from scipy import integrate

from constant import *

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
            return 1
        else :
            return 0
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
            return 1
        else :
            return 0
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