"""
Code with the system for nuclear matter at finite T
"""
import numpy as np

from constant import *
from utilities import *


def system_MS(V, n, p,
            g_sig_NN,  g_omg_NN,  g_rho_NN,
            dg_sig_NN, dg_omg_NN, dg_rho_NN, T):
    """
    system to solve

    Parameters
    ----------
    V : 1darray
        array of the variables
    n : float
        neutron density
    p : float
        proton density
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
    r1 = 0
    r2 = 0
    r3 = 0
    r4 = 0
    r5 = 0
    """
    sig, omg, rho, mu_p, mu_n = V

    #effective masses
    m_eff_p = m_n[0] - sig*g_sig_NN
    m_eff_n = m_n[1] - sig*g_sig_NN
    #termini di energia
    Sigma_p  = g_omg_NN*omg - g_rho_NN*rho
    Sigma_n  = g_omg_NN*omg + g_rho_NN*rho
    Sigma_r  = (dg_omg_NN/g_omg_NN)*(m_m[1]*omg)**2
    Sigma_r += (dg_rho_NN/g_rho_NN)*(m_m[2]*rho)**2
    Sigma_r -= (dg_sig_NN/g_sig_NN)*(m_m[0]*sig)**2
    #numerical integrals for sigma
    I_p = Inte(ns, (m_eff_p, Sigma_p + Sigma_r, mu_p, T))
    I_n = Inte(ns, (m_eff_n, Sigma_n + Sigma_r, mu_n, T))
    #fermi dirac integrals
    I_f_p_p = Inte(fermi_dirac_p, (m_eff_p, Sigma_p + Sigma_r, mu_p, T, True))
    I_f_n_p = Inte(fermi_dirac_p, (m_eff_n, Sigma_n + Sigma_r, mu_n, T, True))
    I_f_p_a = Inte(fermi_dirac_a, (m_eff_p, Sigma_p + Sigma_r, mu_p, T, True))
    I_f_n_a = Inte(fermi_dirac_a, (m_eff_n, Sigma_n + Sigma_r, mu_n, T, True))
    #equation to solve
    r1 = sig - g_sig_NN/m_m[0]**2 * (I_p + I_n)
    r2 = omg - g_omg_NN/m_m[1]**2 * (n + p)
    r3 = rho - g_rho_NN/m_m[2]**2 * (n - p)
    r4 = p - (I_f_p_p - I_f_p_a)
    r5 = n - (I_f_n_p - I_f_n_a)

    return [r1, r2, r3, r4, r5]