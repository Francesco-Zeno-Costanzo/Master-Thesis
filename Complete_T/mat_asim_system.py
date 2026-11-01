"""
Code with three type of siste for beta equilibrium at finite T:

1) system_ele -> neutrons, protons and electrons
2) system_muo -> system_ele + muons
3) system_neu -> system_muo + neutrinos

"""
import numpy as np

from constant import *
from utilities import *

#==============================================================
# neutrons, protons and electrons
#==============================================================


def system_ele(V, nb, g_sg, g_om, g_rh, dg_sg, dg_om, dg_rh, T):
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
    T : float
        temperature

    Returns
    -------
    list of equations to solve
    ri = 0
    """
    sig, omg, rho, p, n, e, mu_p, mu_n, mu_e = V

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
    #fermi dirac integrals electrons
    I_f_e_p = Inte(fermi_dirac_p, (m_el, 0, mu_e, T, True))
    I_f_e_a = Inte(fermi_dirac_a, (m_el, 0, mu_e, T, True))
    #equation to solve
    r1 = sig - g_sg/m_m[0]**2 * (I_p + I_n)
    r2 = omg - g_om/m_m[1]**2 * (n + p)
    r3 = rho - g_rh/m_m[2]**2 * (n - p)
    r4 = nb - n - p
    r5 = p - e
    r6 = p - (I_f_p_p - I_f_p_a)
    r7 = n - (I_f_n_p - I_f_n_a)
    r8 = e - (I_f_e_p - I_f_e_a)
    r9 = mu_n - mu_p - mu_e

    return [r1, r2, r3, r4, r5, r6, r7, r8, r9]


#==============================================================
# neutrons, protons, electrons and muons
#==============================================================


def system_muo(V, nb, g_sg, g_om, g_rh, dg_sg, dg_om, dg_rh, T):
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
    T : float
        temperature

    Returns
    -------
    list of equations to solve
    ri = 0
    """
    sig, omg, rho, p, n, e, mu_p, mu_n, mu_e, m, mu_m = V

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
    I_f_e_p = Inte(fermi_dirac_p, (m_l[0], 0, mu_e, T, True))
    I_f_e_a = Inte(fermi_dirac_a, (m_l[0], 0, mu_e, T, True))
    I_f_m_p = Inte(fermi_dirac_p, (m_l[1], 0, mu_m, T, True))
    I_f_m_a = Inte(fermi_dirac_a, (m_l[1], 0, mu_m, T, True))
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
    r10 = mu_n - mu_p - mu_e
    r11 = mu_e - mu_m

    return [r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11]


#==============================================================
# neutrons, protons, electrons, muons and neutrinos
#==============================================================


def system_neu(V, nb, g_sg, g_om, g_rh, dg_sg, dg_om, dg_rh, T):
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
    T : float
        temperature

    Returns
    -------
    list of equations to solve
    ri = 0
    """
    sig, omg, rho, p, n, e, mu_p, mu_n, mu_e, m, mu_m, nu, mu_nu = V

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
    r12 = mu_e - mu_m
    r13 = nu + e - 0.33*nb

    return [r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13]


